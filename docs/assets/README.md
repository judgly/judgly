# Assets

| file | what |
|---|---|
| `banner.svg` | the README banner (1280 x 640, text only) |
| `social-preview.png` | the same banner as a 1280 x 640 PNG, for GitHub's social preview |
| `how-it-works.svg` | the diagram in the README |

The PNG is rendered from the SVG with
`rsvg-convert -w 1280 -h 640 docs/assets/banner.svg -o docs/assets/social-preview.png`.

GitHub does not read the social preview from the repository: upload `social-preview.png` by
hand under the repository's Settings, General, Social preview.
