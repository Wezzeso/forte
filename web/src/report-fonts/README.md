These TrueType files contain the same glyphs as the existing local UpMax regular
and bold WOFF fonts in `public/zimran/`. They were converted with FontTools by
loading each WOFF, setting `font.flavor = None`, and saving as TTF. jsPDF requires
TrueType for embedded, searchable Cyrillic text. Asset provenance remains in
`public/zimran/SOURCES.md`.
