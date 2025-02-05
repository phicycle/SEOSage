Installation Guide
================

Prerequisites
------------

Before installing SEO Ninja, ensure you have the following prerequisites:

* Python 3.8 or higher
* pip (Python package installer)
* Virtual environment (recommended)
* Git (for cloning the repository)

Required API Keys
---------------

SEO Ninja requires several API keys for full functionality:

* **OpenAI API Key**: For content generation and analysis
* **Google Search Console Credentials**: For SEO data access
* **Moz API Token**: For SEO metrics and analysis

Installation Steps
----------------

1. Clone the Repository
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/yourusername/seoninja.git
   cd seoninja

2. Create Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows

3. Install Dependencies
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install -r requirements.txt

4. Configure Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a `.env` file in the project root:

.. code-block:: bash

   GSC_CLIENT_SECRETS_PATH=client_secrets.json
   MOZ_API_TOKEN=your_moz_token
   TARGET_DOMAIN=your_domain
   OPENAI_API_KEY=your_openai_key

5. Set Up Google Search Console Authentication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Go to Google Cloud Console
2. Create a new project
3. Enable Google Search Console API
4. Create credentials (OAuth 2.0 Client ID)
5. Download the client secrets file
6. Rename it to `client_secrets.json` and place it in the project root

Running the Application
---------------------

1. Start the API Server
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m src.seoninja.api.app

2. Access the API Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open your browser and navigate to:

* API Documentation: `http://localhost:5000/swagger/`
* ReDoc Interface: `http://localhost:5000/redoc/`

Development Setup
---------------

For development, additional steps are recommended:

1. Install Development Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install -r requirements-dev.txt

2. Set Up Pre-commit Hooks
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pre-commit install

3. Run Tests
^^^^^^^^^^^

.. code-block:: bash

   pytest

4. Build Documentation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd docs
   make html

Troubleshooting
--------------

Common Issues
^^^^^^^^^^^^

1. **Authentication Errors**
   
   * Ensure all API keys are correctly set in `.env`
   * Verify Google Search Console credentials
   * Check file permissions for credential files

2. **Import Errors**
   
   * Verify virtual environment is activated
   * Confirm all dependencies are installed
   * Check Python version compatibility

3. **Runtime Errors**
   
   * Check log files in `logs/` directory
   * Verify system resources availability
   * Ensure all required services are running

Getting Help
-----------

If you encounter any issues:

1. Check the :doc:`troubleshooting` guide
2. Search existing GitHub issues
3. Create a new issue with:
   * Error message
   * Steps to reproduce
   * System information
   * Relevant logs 