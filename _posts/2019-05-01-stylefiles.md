---
layout: post
title: Python Style Files
date: 2019-05-01 00:00:00
description: Automatic styling of matplotlib plots
img: style.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Big Bang Nucleosynthesis, Dark Matter]
---

Many of the visual effects that improve the quality of plots in `matplotlib` can be achieved via a user-defined style.

## Style Files

Whilst there are a number of inbuilt styles that can be used via the command,

{% highlight python %}
import matplotlib.pyplot as plt
plt.style.use(style)
{%endhighlight%}

these often lack the details required for scientific publications such as the use of TeX commands, and label sizing. The alternative is to define your own style file. The style file `ja.mplstyle` is the current styling I use for my plots, you can download it <a href="{{site.baseurl}}/assets/files/ja.mplstyle" target="_blank"><i class="fa fa-file-text-o"></i> here</a>. It saves many iterations of the command;

{% highlight python %}
plt.rcparams[...] = ...
{%endhighlight%}

at the start of plotting scripts. To create your own style there are three steps;

1. Create the style file from the template below

2. Work out where on your system to store the file so that `matplotlib` can find it

3. Use the style in your code

## Creating the File


I've included my style file, `ja.mplstyle`, above as an example. This includes many commands that are commented out with a # symbol, but can be set as desired. As a note, of course the file can be named anything you want, it will only change how it is imported later.

## Where to store the file

This is a more complicated question and took some time to figure out. In order for `matplotlib` to find the file and it be included as a possible style, it must be placed in the correct `stylelib` folder. Below there are two options based on my experience with CentOS and OSX, however you may have to be a bit more patient and try some more creative options.

* **OSX:** Place the file in a directory with path `~/.matplotlib/stylelib/ja.mplstyle`
* **CentOS:** Put the file in a directory with path `~/.config/matplotlib/stylelib/ja.mplstyle`

To check if you have placed it in the right folder, the following code can be used. If successful, the name of your style should show up in the list along with the built in styles such as `ggplot` etc.

{%highlight python%}
import matplotlib.pyplot as plt
print(plt.style.available)
{%endhighlight%}

As a slight attempt to troubleshoot this, there is already a directory in the `site-packages` folder where `matplotlib` is contained which is entitled `stylelib`. I found that putting the style file in this directory did not work however as when `matplotlib` was imported, my style was not an option.

## Using the style in your code

Once the style is installed succesfully in the correct directory, using it is simple. At the top of your file, simply include the following, replacing `ja` with the name of your style.

{%highlight python%}   
import matplotlib.pyplot as plt
plt.style.use('ja')
{%endhighlight%}

## Default Styling

As a final, slightly more complicated point, we can also make this the default style so that the second line above is not needed. To do this, add the following line to the file `/anaconda3/lib/site-packages/matplotlib/pyplot .py`,

{%highlight python%}
matplotlib.style.use('ja')
{%endhighlight%}

again, replacing the name as necessary.

## Download

Download my default style file here: <a href="{{site.baseurl}}/assets/files/ja.mplstyle" target="_blank"><i class="fa fa-file-text-o"></i> ja.mplstyle</a>
