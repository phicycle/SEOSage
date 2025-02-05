import os
import sys
from unittest.mock import MagicMock

# Mock modules that might cause issues
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    'google.oauth2.credentials',
    'google_auth_oauthlib.flow',
    'googleapiclient.discovery',
    'google.auth.transport.requests'
]
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'SEO Ninja'
copyright = '2024, Your Name'
author = 'Your Name'
release = '1.0.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinxcontrib.httpdomain',
    'autoapi.extension',
    'myst_parser',
    'sphinxcontrib.openapi',
]

# AutoAPI settings
autoapi_type = 'python'
autoapi_dirs = ['../src']
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]

# Flask AutoAPI settings
autoapi_template_dir = '_templates/autoapi'
autoapi_generate_api_docs = True
autoapi_python_use_implicit_namespaces = True
autoapi_python_class_content = 'both'
autoapi_file_patterns = ['*.py']

# OpenAPI settings
openapi_spec_path = '../openapi.yaml'

# Theme settings
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'both',
    'style_external_links': True,
}

# Other settings
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

def setup(app):
    """Generate OpenAPI spec from Flask routes."""
    try:
        from apispec import APISpec
        from apispec.ext.marshmallow import MarshmallowPlugin
        from flask_apispec.extension import FlaskApiSpec
        from src.seoninja.api.app import app
        
        app.config.update({
            'APISPEC_SPEC': APISpec(
                title='SEO Ninja API',
                version='v1',
                plugins=[MarshmallowPlugin()],
                openapi_version='3.0.2'
            ),
            'APISPEC_SWAGGER_URL': '/swagger/',
        })
        
        docs = FlaskApiSpec(app)
        
        # Generate OpenAPI spec
        with app.test_request_context():
            spec = docs.spec.to_yaml()
            with open(openapi_spec_path, 'w') as f:
                f.write(spec)
    except Exception as e:
        print(f"Warning: Could not generate OpenAPI spec: {e}")
        # Create a minimal OpenAPI spec if generation fails
        minimal_spec = """
openapi: 3.0.2
info:
  title: SEO Ninja API
  version: v1
paths: {}
        """
        with open(openapi_spec_path, 'w') as f:
            f.write(minimal_spec) 