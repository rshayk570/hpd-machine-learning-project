"""Group the 47 `status_description` wordings into outcome classes.

The table has no label column. The outcome of a complaint is only written as free text, and the same
outcome appears under several wordings (some only for certain years, some prefixed with "More than one
complaint was received for this building-wide condition"). `group_of` maps a wording to a class by text
pattern. Audit it against outputs/wording_groups.csv, which lists every wording with its class.

Classes
  P           inspector issued a violation (also: heat inadequate verified via another apartment,
              Section 8 failure)
  N           inspector inspected and found no violation
  N_partial   no violation issued, but the inspection was partial or heat was not legally required
              (outside temperature above 55 F)
  P_existing  violations had already been issued for these conditions (nothing new found)
  P_lead      potential lead-based paint flagged, no violation stated
  U_fixed     a tenant or a phone contact said the condition was corrected (no inspector verdict)
  U_noaccess  no access, inspection not completed, or not enough time (no verdict)
  U_open      complaint still open (no verdict yet)
  U_missing   wording is missing
"""
import pandas as pd

GROUPS = ["P", "N", "N_partial", "P_existing", "P_lead", "U_fixed", "U_noaccess", "U_open", "U_missing"]

# Turning classes into a binary label is a modelling choice. The project's rule: violation issued = 1, inspected
# and clean = 0. None means "excluded from the evaluation population" (no verdict, partial, or already on record).
LABEL_RULE = {
    "P": 1,
    "N": 0,
    "N_partial": None,
    "P_existing": None,
    "P_lead": None,
    "U_fixed": None,
    "U_noaccess": None,
    "U_open": None,
    "U_missing": None,
}


def group_of(text):
    """Map one status_description wording to a class. Returns 'OTHER' for a wording nobody has read yet."""
    if not isinstance(text, str) or not text.strip():
        return "U_missing"
    s = text.lower()
    if "previously issued" in s:
        return "P_existing"
    if "potential lead-based paint" in s and "violations were issued" not in s:
        return "P_lead"
    if (("violations were issued" in s and "no violations were issued" not in s)
            or "a violation was issued" in s or "section 8 failure" in s):
        return "P"
    if "did not violate" in s or "inspected the following conditions. no violations were issued" in s:
        return "N"
    if "heat was not required" in s or "unable to access the rooms" in s:
        return "N_partial"
    if "still open" in s:
        return "U_open"
    if any(k in s for k in ("verified that the following conditions were corrected",
                            "indicated that the condition was corrected",
                            "advised by a tenant",
                            "occupant of the building confirmed")):
        return "U_fixed"
    if any(k in s for k in ("not able to gain access", "unable to complete",
                            "did not have enough time", "was unable to access")):
        return "U_noaccess"
    return "OTHER"


def add_group(df, col="status_description"):
    """Return df with a `group` column; stop loudly if any wording falls outside the known classes."""
    out = df.copy()
    out["group"] = out[col].map(group_of)
    other = out.loc[out["group"] == "OTHER", col].unique()
    if len(other):
        raise ValueError("Unclassified wordings, read them and extend group_of():\n" + "\n".join(map(str, other)))
    return out
