---
title: 'The Namespace Rabbit Hole'
date: 2017-03-30
categories: ["R"]
tags: ["R", "packages"]
author: Matt Cole
---

I've been working on an [R package](https://github.com/mattkcole/FAtools) for factor analysis visualization for a while now, and ran into an interesting problem. One function in [FAtools](https://github.com/mattkcole/FAtools) is essentially a wrapper for several functions in the nFactors package, which plots and displays both graphical and non graphical [scree test](http://econtent.hogrefe.com/doi/abs/10.1027/1614-2241/a000051?journalCode=med) solutions. It's a handy way for people to look *a little* more closely at the number of factors to extract (although, I would argue not enough).

My `scree_plot` function requires two functions from the nFactors package: `parallel` and `nScree`. So, using [roxygen2](https://cran.r-project.org/web/packages/roxygen2/vignettes/roxygen2.html), I added the following to my DESCRIPTION file, telling R that the nFactors package would be utilized in FAtools:

```
Imports:
    nFactors
```

and the following roxygen2 code above my `scree_plot` function to tell R that in this function we would be utilizing `parallel` and `nScree` from nFactors:

```
#' @importFrom nFactors parallel nScree
```

I thought to myself - I'm all set, I can't wait to share this function! But in a few seconds my package dreams were shattered. While building, I received the following error:

```
Error: could not find function "mvrnorm"
```

That's weird. `mvrnorm` is a function from the `MASS` package which is called by `parallel`. `mvrnorm` allows us to simulate random draws from a multivariate normal distribution which `parallel` requires in order to conduct parallel analysis. Clearly, R was able to see `parallel`, which was from the nFactors package, but was not able to find a package it relied on - very strange.

After some head scratching, I thought of an easy fix - explicitly import `mvrnorm`. The following changes were made:

DESCRIPTION:

```
Imports:
    MASS,
    nFactors
```

top of scree_plot.R

```
#' @importFrom MASS mvrnorm
#' @importFrom nFactors parallel nScree

```

Building.

Building..

Building...

Boom, same problem again: `Error: could not find function "mvrnorm"`.

I explicitly called `mvrnorm` into the namespace yet R couldn't find it? That doesn't seem right at all. I was getting worried when the hackerman inside came out. I added the following line *inside* my `scree_plot` function with the hope that by defining `mvrnorm` within `scree_plot`, something would catch.

```
#' ...
#' @importFrom MASS mvrnorm
#' @importFrom nFactors parallel nScree
#' ...
scree_plot <- function(...){
        mvrnorm <- MASS::mvrnorm
        ...
}
```

then:

```
Error: could not find function "mvrnorm"
```

...

Again, disappointment - but also confusion. Phrases like 'lexical scoping' and 'exporting namespace' were swirling around my head as I was trying to figure out what was happening. One last idea I had was to literally call `library(MASS)` _inside_ my function like so:

```
scree_plot <- function(...){
        library(MASS)
        ...
}
```

To be honest, I didn't expect this would work, but alas it allowed `scree_plot` to function as intended. Now there are several reasons why this is a terrible idea. Most importantly, in my own work, I use dplyr's `select` function, which would now be masked by `MASS::select`, other people could have even more conflicts because of the now bloated namespace (MASS is not a small package).

It was around this time I tracked down my friend and R package guru [Sean](http://seankross.com/) to help me see what was going on. After some digging around, it was discovered that, prior to R Version 2.14.0 (released in 2011), the only way for a package function to incorporate functions from other packages was by 'depending' on them\*. Today, the `Depends` field is typically reserved for R version numbers ie. `Depends: R (>= 3.1.0)`, which would restrict package use to R version numbers greater than or equal to 3.1.0. Although usually frowned upon, this field can also specify packages (and versions) that are to be essentially loaded concurrently. As Sean discovered, package functions which incorporate another package's function via *Depends* cannot be incorporated in your packages, or my packages, or anyone else's package without depending on the same package. Because the last update to the nFactors package was also in 2011, we knew the source of our errors, nFactors depends on MASS.

So, in my case, I could not incorporate nFactor's `parallel` function without specifying: `Depends: MASS`. This is not ideal for some of the same reasons calling `library(MASS)` inside of a function is not a good idea. But at the end of the day I have only a few options:

* Use `Depends: MASS` and require all users to have and 'load' MASS when loading FAtools
* Get the maintainers of nFactors to update their code (that has been neglected for six years)
* Build my own versions of relevant nFactor functions like `parallel`

Each of these would solve my problem while potentially creating new ones or redundant work for myself. But, as of now, I plan on doing all of these in that order. Depending on MASS is not ideal, but for now it certainly gets the job done. Emailing the maintainers of nFactors can't hurt, but I'm not particularly hopeful someone would be willing and able to make changes to their package after 6 years of dormancy (although I would be willing to help). Lastly, something I will also probably do, is just write my own functions for conducting parallel analysis so I wouldn't need nFactors, possibly even using c++ ([rcpp](http://www.rcpp.org/)), which is something I'd like to play around with more.

So, what did I learn? Well, a ton of things about R package building, including the many ways (you should be able) to import functions from other packages, best practices, etc. Mainly however, IF you are building a package which draws functions from 'package A' which depends on 'package B', you _MUST_ *depend* on 'package B' and *import* 'package A' (although you could also depend on package A, but that's bad form) to use any functions from 'Package A' that also incorporates 'package B'.

Moral of the story? I'm not sure, but beware of incorporating non-maintained packages in your projects (sage advice).


Relevant links:

* Old package dependency mechanism [source](http://r-pkgs.had.co.nz/description.html)

* R [namespaces](http://r-pkgs.had.co.nz/namespace.html)

* [Hackerman](https://www.youtube.com/watch?v=fQGbXmkSArs&ab_channel=mrfyote)
