from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    path.name: path.read_text(encoding="utf-8")
    for path in [
        ROOT / "frontend" / "index.html",
        ROOT / "frontend" / "calibrate.html",
        ROOT / "frontend" / "depth3d.html",
        ROOT / "phone" / "capture.html",
    ]
}


def test_production_pages_do_not_contain_prototype_tutorial_copy():
    combined = "\n".join(PAGES.values())
    removed_phrases = (
        "Rysowanie: klikaj wierzchołki",
        "Klikaj wierzchołki na podglądzie",
        "Rozłóż markery ArUco na ziemi",
        "Pola oznaczone * są wymagane",
        "Przycisk generuje wysokiej jakości PNG",
        "Zeskanuj kod aparatem telefonu",
        "Kalibracja korzysta z dokładnie tego samego strumienia",
        "Przygotuj markery",
        "Ograniczenie: wynik jest estymacją monokularną",
    )
    for phrase in removed_phrases:
        assert phrase not in combined
