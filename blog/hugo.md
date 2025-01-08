+++
date = '2025-01-06T17:34:21Z'
draft = true
title = 'Hugo: A Static Site Generator'
+++


# Hugo: A Static Site Generator

In this post I give an introduction to what I think is the best static site generator: [Hugo](https://gohugo.io/). 

## What is Hugo?

Hugo is an open-source static site generator written in Go. It takes structured content, often written in Markdown, and compiles it into static HTML, CSS, and JavaScript files. 

---

## Setting Up Hugo: A Quickstart Guide

Follow these steps to set up your first Hugo site

### 1. Install Hugo

First, ensure you have Hugo installed. Use your package manager of choice:

```bash
# Windows
winget install Hugo.Hugo.Extended

```

### 2. Create a New Project

Initialize a new Hugo site:

```bash
hugo new site my-new-site
```

This command scaffolds a directory structure optimized for modular development.

### 3. Add a Theme or Build Your Own

Browse [Hugo Themes](https://themes.gohugo.io/) or create a custom theme to match your project’s needs. For example:

```bash
# change directory to your newly created site
cd my-new-site

# this will insert a template used to display your site
git submodule add https://github.com/vimux/mainroad.git themes/mainroad 
```

That is the theme this site uses. The cool thing about themes is you can quickly change how your site looks.


Set the theme in `config.toml`:

```toml
theme = "mainroad"
```

### 4. Generate Content

Hugo’s Markdown support simplifies content creation. Create a new post:

```bash
hugo new post/first-post.md
```

This will generate a page first-post.md in the post folder with the following contents

```yaml
+++
date = '2025-01-06T17:26:44Z'
draft = true
title = 'First Post'
+++
```

You can edit the generated file in `content/posts`. 

### 5. Local Development Server

Hugo’s built-in server lets you preview your site with hot reloading:

```bash
hugo server -D
```

Access your site at `http://localhost:1313`.

This means you can view your change live it can help you write content quickly

### 6. Build Static the files

When your site is ready for deployment, run:

```bash
hugo

```

Static files will be placed in the `public` directory, ready for deployment.

---

## Deploying Hugo Sites

Static sites are highly portable and can be deployed across various platforms. Here are some examples:

1. **AWS S3**: Use S3 buckets for scalable hosting, optionally with CloudFront for CDN.
2. **Github Pages**: This is a brilliant free way to deploy a website.
---

## Advanced Tips for Programmers

- **Shortcodes**: Define reusable snippets for embedding code, videos, or custom widgets.

### Embed a youtube video


```
{{< youtube id="dQw4w9WgXcQ" >}}
```

### Embed an image with optional attributes 

```
{{< figure src="/img/cliffs.jpg">}}
```


- **Partials**: Modularize your templates for clean and maintainable code.
- **Data Files**: Use YAML, JSON, or TOML files to inject structured data into your templates.
- **Hugo Modules**: Enable component-based development by reusing content and templates across projects.



- **Benchmarking**: Use `hugo --templateMetrics` to optimize template rendering.

---

## Useful parameters

In the hugo.toml file we put configuration options for the generate site for example 

googleAnalytics: if you have set uip a google analytics account add this property and it will generate the correct addition to your site.


## Further reading

[Quick Start Guide](https://gohugo.io/getting-started/quick-start/)  
[How to host on github pages](https://gohugo.io/hosting-and-deployment/hosting-on-github/)  
[Hugo Discord](https://discourse.gohugo.io)  
[Hugo Tips](https://discourse.gohugo.io/t/hugo-tutorials-tips)  
[Markdown Tutorial](https://commonmark.org/help/tutorial/)  

