// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!
#let project(title: "", abstract: [], authors: (), body) = {
  // Set the document's basic properties.
  set document(author: authors, title: title)
  set page(numbering: "1", number-align: center)
  set text(font: "New Computer Modern", lang: "en")
  show math.equation: set text(weight: 400)
  set heading(numbering: "1.1.")

  // Set paragraph spacing.
  show par: set block(above: 1.2em, below: 1.2em)
  set par(leading: 0.75em, justify: true)

  // // Set run-in subheadings, starting at level 3.
  // Increase top-padding of level-2+ headings
  show heading: it => {
    if it.level > 1 {
      pad(top: 0.5em, it)
    }
    else {
      it
    }
  }
  /* show heading: it => {
    if it.level > 2 {
      parbreak()
      text(11pt, style: "italic", weight: "regular", it.body + ".")
    } else {
      it
    }
  } */


  // Title row.
  align(center)[
    #block(text(weight: 700, 1.7em, title))
  ]

  // Author information.
  pad(
    top: 0.5em,
    x: 2em,
    grid(
      columns: (1fr,) * calc.min(3, authors.len()),
      gutter: 1em,
      ..authors.map(author => align(center, strong(author))),
    ),
  )

  // Abstract.
  pad(
    x: 2em,
    top: 1em,
    bottom: 2em,
    align(center)[
      #heading(
        outlined: false,
        numbering: none,
        text(0.85em, smallcaps[Abstract]),
      )
      #abstract
    ],
  )

  // Main body.
  set par(justify: true)
  //show: columns.with(2, gutter: 1.3em)

  body
}