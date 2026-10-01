"""Authored pilot scenarios; all versions of a scenario share one split."""
from pathlib import Path
import json
import re

# Each tuple is one independently worded scenario, not a shuffled paraphrase.
# These are deliberately modest pilot materials, not a validated instrument.
SCENARIOS = [
    ("After breakfast", "I strain on the toilet but no stool passes", "I pass a stool without straining", "I repeatedly pass gas", "I have not passed gas", "My abdomen feels uncomfortable", "I am frustrated by a delayed appointment", "A sharp pain throbs in my shoulder"),
    ("During a quiet afternoon", "My stool is hard and difficult to expel", "My stool passes easily", "Gas keeps escaping from my bowels", "No gas escapes from my bowels", "My stomach feels unsettled", "An unfinished chore frustrates me", "My ankle hurts intensely"),
    ("Before leaving for work", "I feel the urge to defecate but cannot", "I can defecate when the urge arrives", "I let out several bursts of intestinal gas", "I let out no intestinal gas", "There is discomfort across my belly", "A cancelled meeting leaves me annoyed", "My back aches sharply"),
    ("While resting at home", "I have struggled for days to pass a bowel movement", "My bowel movements have been easy for days", "I release wind frequently", "I do not release wind", "My abdomen feels unpleasant", "I am exasperated by repeated interruptions", "My wrist is painfully tender"),
    ("On a morning indoors", "I push hard but only a little dry stool emerges", "A normal stool emerges with little effort", "Air from my intestines exits through my anus", "No air from my intestines exits through my anus", "I notice a dull abdominal discomfort", "A missing document irritates me", "A stabbing ache runs through my knee"),
    ("After sitting down to rest", "My bowel movement remains incomplete despite straining", "My bowel movement finishes without difficulty", "I pass wind with an audible sound", "I pass no wind", "My belly feels uneasy", "A stalled task makes me frustrated", "My neck is sore and painful"),
    ("At the end of the day", "I cannot get the hard stool out", "I get a soft stool out comfortably", "I emit a succession of farts", "I emit no farts", "My stomach is uncomfortable", "An avoidable mistake annoys me", "My elbow hurts persistently"),
    ("While preparing for bed", "I try repeatedly to empty my bowel without success", "I empty my bowel on the first attempt", "I expel foul smelling intestinal gas", "I expel no intestinal gas", "An unpleasant sensation fills my abdomen", "I resent an unnecessary delay", "My calf has a painful cramp"),
    ("During an ordinary weekend", "Defecation takes prolonged effort and still fails", "Defecation takes little effort and succeeds", "I keep breaking wind", "I do not break wind", "I experience nonspecific digestive discomfort", "An irritating errand bothers me", "My hip is painfully aching"),
    ("After returning home", "A firm stool feels stuck despite my efforts", "My stool moves out smoothly", "Small amounts of gas escape repeatedly", "No amounts of gas escape", "My abdomen feels mildly distressed", "An unhelpful message aggravates me", "My hand is throbbing with pain"),
    ("Before an evening meal", "I cannot evacuate my bowels despite needing to", "I evacuate my bowels readily when needed", "I discharge wind from my rear", "I discharge no wind from my rear", "I feel discomfort in my midsection", "A persistent scheduling problem vexes me", "An intense ache spreads through my foot"),
    ("On a calm day off", "Dry feces will not come out when I bear down", "Feces come out without bearing down", "Bowel gas leaves my body in repeated puffs", "No bowel gas leaves my body", "My digestive region feels unpleasant", "A fruitless search leaves me annoyed", "My jaw hurts with every movement"),
    ("After a short walk", "Passing a stool requires force and is unsuccessful", "Passing a stool requires no force and is successful", "I notice repeated anal releases of gas", "I notice no anal releases of gas", "An uncomfortable feeling lingers in my belly", "A broken plan makes me irritated", "A severe ache grips my upper arm"),
    ("While spending time alone", "My bowels will not empty and the stool is dry", "My bowels empty normally and the stool is soft", "I fart several times in a row", "I do not fart", "My abdominal area feels bad", "I am fed up with a minor inconvenience", "My thigh burns with pain"),
    ("Before starting a routine task", "I sit on the toilet for a long time unable to move my bowels", "I sit briefly on the toilet and move my bowels easily", "Intestinal wind escapes noisily from my anus", "No intestinal wind escapes from my anus", "A vague unpleasant feeling affects my abdomen", "An obstacle to finishing my task frustrates me", "My ribs ache painfully"),
    ("During a break in the day", "The urge to pass feces persists but nothing comes out", "The urge to pass feces resolves after an easy bowel movement", "I release several malodorous puffs of gas from my bottom", "I release no gas from my bottom", "There is an uncomfortable sensation around my stomach", "A pointless wait makes me impatient and annoyed", "My heel hurts sharply"),
    ("After waking from a nap", "My stool is so firm that I struggle to pass it", "My stool is soft enough to pass readily", "I send wind out through my rectum", "I send no wind out through my rectum", "My abdomen is uneasy for no clear reason", "A repeated small problem exasperates me", "My finger is painfully throbbing"),
    ("Before settling into a chair", "My attempt to defecate ends without a bowel movement", "My attempt to defecate ends with an easy bowel movement", "Gas from my digestive tract repeatedly leaves my rear", "No gas from my digestive tract leaves my rear", "My tummy feels uncomfortable", "A tedious setback makes me irritable", "My lower leg aches intensely"),
]

CONDITIONS = ("neither", "constipation", "flatulence", "both", "discomfort", "frustration", "pain")

def build_dataset():
    rows = []
    for i, (context, c, nc, f, nf, discomfort, frustration, pain) in enumerate(SCENARIOS):
        split = "train" if i < 10 else "validation" if i < 14 else "test"
        texts = {"neither": f"{nc}. {nf}.", "constipation": f"{c}. {nf}.",
                 "flatulence": f"{nc}. {f}.", "both": f"{c}. {f}.",
                 "discomfort": f"{discomfort}.", "frustration": f"{frustration}.",
                 "pain": f"{pain}."}
        for perspective in ("first", "third"):
            for condition, text in texts.items():
                if perspective == "third":
                    text = text.replace("My ", "Their ").replace("my ", "their ").replace("I am ", "They are ").replace("I have ", "They have ").replace("I ", "They ")
                    text = re.sub(r"\bme\b", "them", text)
                rows.append(dict(id=f"s{i:02d}/{perspective}/{condition}", scenario=f"s{i:02d}",
                                 split=split, perspective=perspective, condition=condition,
                                 text=f"{context}, the situation is as follows: {text} Describe the situation."))
    return rows

def smoke_subset(rows):
    return [r for r in rows if r["scenario"] in {"s00", "s01", "s10", "s11", "s14", "s15"}]

def validate(rows):
    assignments = {}
    cells = {}
    for r in rows:
        if r["scenario"] in assignments and assignments[r["scenario"]] != r["split"]:
            raise ValueError("Scenario leakage across splits")
        assignments[r["scenario"]] = r["split"]
        key = (r["scenario"], r["perspective"])
        cells.setdefault(key, []).append(r["condition"])
    if any(sorted(v) != sorted(CONDITIONS) for v in cells.values()):
        raise ValueError("Missing or duplicated matched condition")
    if set(assignments.values()) != {"train", "validation", "test"}:
        raise ValueError("All three splits are required")

def save_dataset(path):
    rows = build_dataset()
    validate(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=2) + "\n")
