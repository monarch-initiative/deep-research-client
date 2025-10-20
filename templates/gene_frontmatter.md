---
format: jinja
description: Gene research template with Jinja2 features
---
Research the {{gene}} gene{% if organism %} in {{organism}}{% endif %}.

Provide information about:
{% for topic in topics %}
- {{topic}}
{% endfor %}
