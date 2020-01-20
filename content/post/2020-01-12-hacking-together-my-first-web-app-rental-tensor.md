---
title: Hacking together my first web app in Django
author: Matt Cole
date: '2020-01-20'
slug: hacking-together-my-first-web-app-Django
categories:
  - web app
tags:
  - building
  - learning
---

### Background

A few months ago I started my journey to build my first web app, [Rental Tensor](http://rentaltensor.com/) (excuse the horrible name 😎) in order to estimate the rents for different properties at the zip code level. As a trained statistician disguised as a techy Data Scientist, I love playing with a mix of code and data, but I feel that it can sometimes be difficult for models or data pipelines to be very valuable without some sort of web backbone to host and serve it. As the old proverb says: "a great model stuck on your hard drive is just some code". While I am familiar with [Shiny](https://shiny.rstudio.com/) which can be used to serve models to real users via a web format (without leaving R!), I felt like it was time for me to learn how to host a more robust and flexible service.

### Tools

I was looking around at all of the different frameworks to make dynamic web applications and felt _extremely_ overwhelmed. Ruby on Rails seemed like a clear 'top' choice used by Airbnb, GitHub, and Hulu, while some companies were using all JavaScript stacks (?!). I guess PHP is still a thing that some people love, and kept reading about so many other options that I had never previously heard of. As I started to feel the grip of decision paralysis wrap it's cold, dead hands around my ankles I started to think about what would be most useful for _me_ and if my decision really mattered at all. 

Surely, _Ruby_ and _Python_ and _JavaScript_ could build similarly useful things with similar amounts of effort. Since I regularly use Python for work and fun, _and_ it is used in so many data science applications, I narrowed my search to just Python web frameworks.

Now that I removed 90% of my options, I only had to choose between two options: Flask and Django. Since the consensus of random people on the internet seemed to be that 'Django makes more decisions for you than Flask' I decided to try to learn Django since I'm not a web developer and don't want to make every decision I might be tempted to make. It might be strange, since almost all blog posts I read a super opinionated - but I felt very comfortable picking either framework and have no idea what I gained or lost by picking Django over Flask. That being said, I'm super glad I didn't need to learn a whole new language just to build something cool. 


### Learning Django

Learning Django (enough to put together something somewhat functional) consisted of following along three tutorials before hacking away: the classic [Django Polls tutorial](https://docs.djangoproject.com/en/3.0/intro/), [Tango with Django](http://leanpub.com/tangowithdjango2/), and [Django for Beginners](https://djangoforbeginners.com/introduction/). 

The [Django Polls tutorial](https://docs.djangoproject.com/en/3.0/intro/) was a great introduction that opened my mind to what was possible with Django. You take a concept (a website where people can vote on questions) and slowly build the parts needed to turn it into a real working website. It really shepherds you through the basic concepts and even shows you some of the internal mechanics that power the Django model/database framework (using interactive SQL) - very cool! This tutorial trades off some completeness in order to be concise and easy to follow, but I definitely recommend this canonical starting place for beginners. Plus it's free!

[Tango with Django](http://leanpub.com/tangowithdjango2/) is a much more in-depth resource which takes you through the basic model (data), view (dynamic aspect), template (display) architectural concepts while guiding you through slightly difficult to understand (for me) topics like user authentication and AJAX requests in a way that helped me actually understand them.  There is also discussion of deployment options and strategies which is actually a little light on details for what I ended up needing. In fact, the issue of properly deploying a 'full fledged application' was, in my experience, the toughest part of developing my first web app. Everything was working fine on my local machine, but there were so many problems getting the static files to load, setting up a database, or even just simply getting the site to render at all on a custom domain. There were a few other [blog posts](https://medium.com/agatha-codes/9-straightforward-steps-for-deploying-your-django-app-with-heroku-82b952652fb4) that really helped in this regard that I will definitely be referring to in the future. The icing on the cake for me in this book was that it actually spends a bit of time explaining different web technologies that you may need to create a particular site (such as JavaScript and bootstrap to make your site look great). With just pure Django it would be hard to have your site look particularly great.

[Django for Beginners](https://djangoforbeginners.com/introduction/) was pretty neat as it was a collection of several small projects that you build from scratch. While much of the material was actually covered in the previous two resources, this book really solidified some key concepts for me and got me feeling more comfortable in the Django environment. 


### My first project

I wanted to make a dynamic calculator that would estimate the rent for a property via scraping real rental listings given a set of property parameters such as location and number of bathrooms. I figured this would be good first project as it was simple (the web app only does one thing - estimate rents), dynamic (we are computing estimated rents on the fly), and the application would scrape rental listings from the web (This wasn't something I had known how to do before). After building a web scraper using [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) I slid it into a Django view and voilà! - I have a working app!

The most difficult hurdles past this point was creating a query string so one could share their results with just a URL, and hosting the web app effectively. The query string issue was something I hadn't thought about before, but it seems very important for as much information as possible to be captured in the 

While I'm excited to have completed my first semi-functional web app, I'm even more glad I have the confidence now to try different things and to incorporate some of my data experience into apps that will hopefully help others. 

