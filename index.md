---
layout: default
title: Blog
---

Welcome to my personal website, where I write about all things technical.

## Articles

{% if site.posts.size > 0 %}
<ul>
  {% for post in site.posts %}
    <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <span style="color:#999; font-size:0.85em;">{{ post.date | date: "%B %-d, %Y" }}</span></li>
  {% endfor %}
</ul>
{% else %}
*No posts yet — check back soon.*
{% endif %}
