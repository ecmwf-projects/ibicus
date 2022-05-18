
****************************
python-package-template-repo
****************************

A template documentation tree for creating readthedocs compatible documentation.

**Quick start**

Follow these steps to prepare your documentation for readthedocs.

#. Install requirements

     pip3 install -r requirements.txt

#. Create the supporting rst files and add them to the source directory, you should remove any
existing rst files to ensure a clean build:

     rm source/reference/*rst
     sphinx-apidoc -o source/reference -H "API reference" --tocfile api -f ../PACKAGE_NAME/

#. Update the source/index.rst to reference the rst files create above. Depending on the contents of your package and the 
rst files produced you will need to add something like the following to the Reference section:

     * :doc:`modules`

     .. toctree::
        :maxdepth: 1
        :hidden:
        :caption: Reference

        modules


These steps will allow readthedocs to construct your documentation pages. It is possible to build the html pages locally 
for testing. From the `docs/source` directory execute the following:

     make html
     open build/html/index.html  # To open with your default html application



    

