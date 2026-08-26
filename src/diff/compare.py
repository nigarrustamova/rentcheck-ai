"""Compare a check-out photo set with a check-in set and report only new damage.

The product question is not "is there damage" — a rental car almost always has
some — but "is there damage now that was not there when the customer drove off".

Pixel-level differencing cannot answer that. Two inspections happen hours or days
apart, by different staff, in different light, with the car parked somewhere else;
the frames never align. So the comparison runs on damage *inventories* instead:
each photo yields a list of damages with a class and a location in normalised
image coordinates, and a damage in the check-in set counts as new when nothing of
the same class sits near it in the matching check-out view.

Every new damage carries a confidence band. The middle band exists because of a
documented real-world failure: rental customers have been billed for damage an
automated scanner flagged wrongly. A system that says "I am not sure, look at this
one" is more useful to an operator than one that silently guesses, and the cost of
that caution is measurable.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The six walk-around positions the capture protocol prescribes. Comparing a
# front-left photo against a rear view would be meaningless, so views are matched
# by name and anything unrecognised is compared only against its own name.
STANDARD_VIEWS = ("front-left", "left", "rear-left", "rear-right", "right", "front-right")

# A detection below REVIEW_CONFIDENCE is dropped entirely; between REVIEW and
# CONFIRMED it is surfaced for a human to check; above CONFIRMED it is reported as
# new damage. These are starting points — tune them on the paired test set and
# report the operating point actually used.
CONFIRMED_CONFIDENCE = 0.50
REVIEW_CONFIDENCE = 0.25

# How close two damages must be to count as the same one across a pair of photos.
# The camera moves between sessions, so a scratch shifts across the frame even when
# nothing about the car changed; too strict a threshold invents new damage, which is
# the failure mode that bills customers for damage they did not cause. Too loose and
# genuinely new damage gets absorbed into an old one and never reported.
#
# These two numbers are the main tunables in this module. The defaults below are
# starting points, not findings — calibrate them on the paired test set and report
# the values actually used.
MATCH_IOU = 0.15
MATCH_CENTROID_DISTANCE = 0.20  # Euclidean distance in normalised xy, where the
                                # full diagonal is sqrt(2), so this is ~14% of it


@dataclass
class Damage:
    """One detected damage instance, in coordinates independent of image size."""

    view: str
    class_name: str
    confidence: float
    area_fraction: float           # mask area as a fraction of the image
    bbox: tuple                    # (x1, y1, x2, y2), normalised to [0, 1]
    area_px: int = 0               # raw mask pixels, kept for the cm2 conversion
    area_cm2: float | None = None  # filled in only when a scale reference is found

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def band(self):
        return "confirmed" if self.confidence >= CONFIRMED_CONFIDENCE else "review"

    def to_dict(self):
        data = asdict(self)
        data["centroid"] = self.centroid
        data["band"] = self.band
        return data


@dataclass
class ViewDiff:
    """The outcome of comparing one view between the two sessions."""

    view: str
    new_damage: list = field(default_factory=list)
    pre_existing: list = field(default_factory=list)
    repaired: list = field(default_factory=list)  # seen at check-out, gone at check-in

    @property
    def confirmed(self):
        return [d for d in self.new_damage if d.band == "confirmed"]

    @property
    def needs_review(self):
        return [d for d in self.new_damage if d.band == "review"]


@dataclass
class DiffReport:
    views: list = field(default_factory=list)
    missing_views: list = field(default_factory=list)  # present in one session only

    @property
    def confirmed(self):
        return [d for v in self.views for d in v.confirmed]

    @property
    def needs_review(self):
        return [d for v in self.views for d in v.needs_review]

    @property
    def pre_existing(self):
        return [d for v in self.views for d in v.pre_existing]

    def summary(self):
        return {
            "new_damage_confirmed": len(self.confirmed),
            "new_damage_needs_review": len(self.needs_review),
            "pre_existing": len(self.pre_existing),
            "views_compared": len(self.views),
            "views_missing": self.missing_views,
            "total_new_area_fraction": round(sum(d.area_fraction for d in self.confirmed), 5),
        }


def view_from_filename(path):
    """Read the view out of a name like car01_s1_front-left.jpg.

    The trailing underscore-separated field is checked first, because a plain
    suffix test would read "rear-left" as "left" and silently collapse two
    different sides of the car into one — which would then compare the rear of the
    vehicle against its flank. Longest-first matching covers names that do not use
    the underscore convention.

    Anything unrecognised falls back to the whole stem, so ad-hoc photo sets — a
    repair shop's before and after pictures, say — still compare against their own
    counterpart rather than being dropped.
    """
    stem = Path(path).stem
    parts = stem.split("_")

    # <car>_<session>_<view>: the last field is the view, whatever it is called.
    # Taking it verbatim rather than only accepting the six standard names is what
    # lets a repair shop's "car07_s1_bumper" pair with "car07_s2_bumper" — their
    # photos are of one panel, not of a walk-around position.
    if len(parts) >= 3:
        return parts[-1]

    for view in sorted(STANDARD_VIEWS, key=len, reverse=True):
        if stem.endswith(view):
            return view
    return stem


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def centroid_distance(a, b):
    (ax, ay), (bx, by) = a.centroid, b.centroid
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def same_damage(a, b):
    """Whether two detections describe the same physical damage.

    Either they overlap, or — when the camera has moved enough that the boxes no
    longer intersect — their centres are close and the class agrees.
    """
    if a.class_name != b.class_name:
        return False
    if iou(a.bbox, b.bbox) >= MATCH_IOU:
        return True
    return centroid_distance(a, b) <= MATCH_CENTROID_DISTANCE


def match_view(before, after):
    """Greedy one-to-one matching between the two sessions for a single view.

    Confident detections are matched first, so a strong check-in detection claims
    its counterpart before a marginal one can take it.
    """
    unmatched_before = list(before)
    new_damage, pre_existing = [], []

    for candidate in sorted(after, key=lambda d: -d.confidence):
        best, best_score = None, 0.0
        for earlier in unmatched_before:
            if not same_damage(candidate, earlier):
                continue
            # Prefer overlap; fall back to proximity so the closest one wins.
            score = iou(candidate.bbox, earlier.bbox) or (
                1.0 - centroid_distance(candidate, earlier)
            )
            if score > best_score:
                best, best_score = earlier, score
        if best is None:
            new_damage.append(candidate)
        else:
            unmatched_before.remove(best)
            pre_existing.append(candidate)

    return new_damage, pre_existing, unmatched_before


def compare(before_by_view, after_by_view):
    """Compare two sessions, each a mapping of view name to a list of Damage."""
    report = DiffReport()

    for view in sorted(set(before_by_view) | set(after_by_view)):
        before = before_by_view.get(view)
        after = after_by_view.get(view)

        if before is None or after is None:
            # A view photographed only once proves nothing either way: we cannot
            # claim damage is new if that side was never seen at check-out.
            report.missing_views.append(view)
            continue

        new_damage, pre_existing, repaired = match_view(before, after)
        report.views.append(
            ViewDiff(
                view=view,
                new_damage=new_damage,
                pre_existing=pre_existing,
                repaired=repaired,
            )
        )

    return report
