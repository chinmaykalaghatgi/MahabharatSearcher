"""Canonical book-number → parva-name mapping.

The Mahabharata's 18 books are the 18 major parvas. Several Layer 1
build scripts label their by-book coverage tables with these names;
this module is the single source of truth so the mapping isn't
duplicated (and can't drift) across modules.
"""

BOOK_NAMES = {
    1: "Adi", 2: "Sabha", 3: "Vana", 4: "Virata", 5: "Udyoga",
    6: "Bhishma", 7: "Drona", 8: "Karna", 9: "Shalya", 10: "Sauptika",
    11: "Stri", 12: "Shanti", 13: "Anushasana", 14: "Ashvamedha",
    15: "Ashramavasika", 16: "Mausala", 17: "Mahaprasthanika",
    18: "Svargarohana",
}


def parva_name(book: int) -> str:
    """Parva name for a book number, or '?' if out of range."""
    return BOOK_NAMES.get(book, "?")
