from jinja2 import DictLoader
import os
import os.path
from traitlets.config import Config
from nbconvert.exporters.html import HTMLExporter


dl = DictLoader(
    {
        'footer': """
{%- extends 'base.html.j2' -%}
{% from 'mathjax.html.j2' import mathjax %}
{% from 'jupyter_widgets.html.j2' import jupyter_widgets %}
{%- block header -%}

{%- block html_head -%}
{% set nb_title = nb.metadata.get('title', resources['metadata']['name']) | escape_html_keep_quotes %}
---
layout: post
title:  {{nb_title}}
date:   2023-05-08 13:07:37 -0500
categories: ml dl
tags: transformers,ml,dl,llm,deep-learning,machine-learning
---

<script src="{{ resources.require_js_url }}"></script>

{% block jupyter_widgets %}
  {%- if "widgets" in nb.metadata -%}
    {{ jupyter_widgets(resources.jupyter_widgets_base_url,
                       resources.html_manager_semver_range,
                       resources.widget_renderer_url) }}
  {%- endif -%}
{% endblock jupyter_widgets %}

{% for css in resources.inlining.css -%}
  <style type="text/css">
    {{ css }}
  </style>
{% endfor %}

{% block notebook_css %}
{{ resources.include_css("static/index.css") }}
{% if resources.theme == 'dark' %}
    {{ resources.include_css("static/theme-dark.css") }}
{% elif resources.theme == 'light'  %}
    {{ resources.include_css("static/theme-light.css") }}
{% else %}
    {{ resources.include_lab_theme(resources.theme) }}
{% endif %}
<style type="text/css">
/* Force rendering true colors when outputing to pdf */
* {
  -webkit-print-color-adjust: exact;
}

/* Misc */
a.anchor-link {
  display: none;
}

.highlight  {
  margin: 0.4em;
}

/* Input area styling */
.jp-InputArea {
  overflow: hidden;
}

.jp-InputArea-editor {
  overflow: hidden;
}

.CodeMirror pre {
  margin: 0;
  padding: 0;
}

/* Using table instead of flexbox so that we can use break-inside property */
/* CSS rules under this comment should not be required anymore after we move to the JupyterLab 4.0 CSS */


.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  min-width: calc(
    var(--jp-cell-prompt-width) - var(--jp-private-cell-scrolling-output-offset)
  );
}

.jp-OutputArea-child {
  display: table;
  width: 100%;
}

.jp-OutputPrompt {
  display: table-cell;
  vertical-align: top;
  min-width: var(--jp-cell-prompt-width);
}

body[data-format='mobile'] .jp-OutputPrompt {
  display: table-row;
}

.jp-OutputArea-output {
  display: table-cell;
  width: 100%;
}

body[data-format='mobile'] .jp-OutputArea-child .jp-OutputArea-output {
  display: table-row;
}

.jp-OutputArea-output.jp-OutputArea-executeResult {
  width: 100%;
}

/* Hiding the collapser by default */
.jp-Collapser {
  display: none;
}

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }

  .jp-OutputArea-child {
    break-inside: avoid-page;
  }
}
</style>

{% endblock notebook_css %}

{%- block html_head_js_mathjax -%}
{{ mathjax(resources.mathjax_url) }}
{%- endblock html_head_js_mathjax -%}

{%- block html_head_css -%}
{%- endblock html_head_css -%}

{%- endblock html_head -%}
{%- endblock header -%}

{% block footer %}
FOOOOOOOOTEEEEER
{% endblock footer %}
"""
    }
)
# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------


class BlogExporter(HTMLExporter):
    """
    Jekyll blog exporter
    """
    export_from_notebook = "blog"
    extra_loaders = [dl]
    template_file = 'footer'

    def _file_extension_default(self):
        """
        The new file extension is ``.test_ext``
        """
        return '.test_ext'

    # @property
    # def template_paths(self):
    #     """
    #     We want to inherit from HTML template, and have template under
    #     ``./templates/`` so append it to the search path. (see next section)

    #     Note: nbconvert 6.0 changed ``template_path`` to ``template_paths``
    #     """
    #     tpl_path = os.path.join(os.path.dirname(__file__), "templates")
    #     print('--------- tpl_path', tpl_path)
    #     print('--------- super()',
    #           [x for x in dir(super()) if not x.startswith('_')])
    #     return super().template_paths + [tpl_path]

    def _template_file_default(self):
        """
        We want to use the new template we ship with our library.
        """
        return 'test_template'  # full
