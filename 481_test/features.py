import re

def extract_features(text):
    text = text.lower()
    feats = []

    # Punctuation check (too many ! or ? is suspicious)
    if "!" in text or "?" in text:
        feats.append("punct_bad")

    # Fake keywords
    fake_keywords = [
        "fake", "hoax", "untrue", "scam", "fraud", "false",
        "misleading", "clickbait", "conspiracy", "bogus", "rumor"
    ]
    if any(word in text for word in fake_keywords):
        feats.append("fake_kw")

    # Real / academic keywords
    academic_high = ["study", "research", "experiment", "dataset", "analysis", "methodology"]
    academic_medium = ["paper", "review", "results", "conclusion"]
    
    if any(word in text for word in academic_high):
        feats.append("academic_high")
    elif any(word in text for word in academic_medium):
        feats.append("academic_medium")
    else:
        feats.append("academic_low")

    # Technical keywords
    technical_high = ["algorithm", "network", "simulation", "optimization", "model", "tensorflow", "pytorch"]
    technical_medium = ["code", "script", "function", "program", "library"]
    
    if any(word in text for word in technical_high):
        feats.append("technical_high")
    elif any(word in text for word in technical_medium):
        feats.append("technical_medium")
    else:
        feats.append("technical_low")

    # Real keywords for credibility
    real_keywords = [
        "official", "report", "confirmed", "ieee", "review",
        "research", "study", "paper", "experiment", "dataset",
        "analysis", "methodology", "results", "conclusion"
    ]
    if any(word in text for word in real_keywords):
        feats.append("real_kw")

    # Spelling heuristic: too many short words may indicate errors
    if len(re.findall(r"\b[a-z]{1,2}\b", text)) > 5:
        feats.append("spell_bad")

    return feats
