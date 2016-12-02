We welcome you to [check the existing issues](https://github.com/EpistasisLab/scikit-rebate/issues/) for bugs or enhancements to work on. If you have an idea for an extension to scikit-rebate, please [file a new issue](https://github.com/EpistasisLab/scikit-rebate/issues//new) so we can discuss it.

## Project layout

The latest stable release of scikit-rebate is on the [master branch](https://github.com/EpistasisLab/scikit-rebate/tree/master), whereas the latest version of scikit-rebate in development is on the [development branch](https://github.com/EpistasisLab/scikit-rebate/tree/development). Make sure you are looking at and working on the correct branch if you're looking to contribute code.

In terms of directory structure:

* All of scikit-rebate's code sources are in the `skrebate` directory
* The documentation sources are in the `docs_sources` directory
* The latest documentation build is in the `docs` directory
* Unit tests for scikit-rebate are in the `tests.py` file

Make sure to familiarize yourself with the project layout before making any major contributions, and especially make sure to send all code changes to the `development` branch.

## How to contribute

The preferred way to contribute to scikit-rebate is to fork the 
[main repository](https://github.com/EpistasisLab/scikit-rebate/) on
GitHub:

1. Fork the [project repository](https://github.com/EpistasisLab/scikit-rebate/):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

          $ git clone git@github.com:YourLogin/scikit-rebate.git
          $ cd scikit-rebate

3. Create a branch to hold your changes:

          $ git checkout -b my-contribution

4. Make sure your local environment is setup correctly for development. Installation instructions are almost identical to [the user instructions](installing.md) except that scikit-rebate should *not* be installed. If you have scikit-rebate installed on your computer, then make sure you are using a virtual environment that does not have scikit-rebate installed. Furthermore, you should make sure you have installed the `nose` package into your development environment so that you can test changes locally.

          $ conda install nose

5. Start making changes on your newly created branch, remembering to never work on the ``master`` branch! Work on this copy on your computer using Git to do the version control.

6. Once some changes are saved locally, you can use your tweaked version of scikit-rebate by navigating to the project's base directory and running scikit-rebate in a script. You can use an example script in our [examples directory](examples/GAMETES_Example.md) to begin your testing.

7. To check your changes haven't broken any existing tests and to check new tests you've added pass run the following (note, you must have the `nose` package installed within your dev environment for this to work):

          $ nosetests -s -v

8. When you're done editing and local testing, run:

          $ git add modified_files
          $ git commit

   to record your changes in Git, then push them to GitHub with:

          $ git push -u origin my-contribution

Finally, go to the web page of your fork of the scikit-rebate repo, and click 'Pull Request' (PR) to send your changes to the maintainers for review. Make sure that you send your PR to the `development` branch, as the `master` branch is reserved for the latest stable release. This will start the CI server to check all the project's unit tests run and send an email to the maintainers.

(If any of the above seems like magic to you, then look up the 
[Git documentation](http://git-scm.com/documentation) on the web.)

## Before submitting your pull request

Before you submit a pull request for your contribution, please work through this checklist to make sure that you have done everything necessary so we can efficiently review and accept your changes.

If your contribution changes scikit-rebate in any way:

* Update the [documentation](https://github.com/EpistasisLab/scikit-rebate/tree/master/docs_sources) so all of your changes are reflected there.

* Update the [README](https://github.com/EpistasisLab/scikit-rebate/blob/master/README.md) if anything there has changed.

If your contribution involves any code changes:

* Update the [project unit tests](https://github.com/EpistasisLab/scikit-rebate/blob/master/tests.py) to test your code changes.

* Make sure that your code is properly commented with [docstrings](https://www.python.org/dev/peps/pep-0257/) and comments explaining your rationale behind non-obvious coding practices.

If your contribution requires a new library dependency:

* Double-check that the new dependency is easy to install via `pip` or Anaconda and supports both Python 2 and 3. If the dependency requires a complicated installation, then we most likely won't merge your changes because we want to keep scikit-rebate easy to install.

* Add a line to pip install the library to [.travis_install.sh](https://github.com/EpistasisLab/scikit-rebate/blob/master/ci/.travis_install.sh#L46)

* Add a line to print the version of the library to [.travis_install.sh](https://github.com/EpistasisLab/scikit-rebate/blob/master/ci/.travis_install.sh#L56)

* Similarly add a line to print the version of the library to [.travis_test.sh](https://github.com/EpistasisLab/scikit-rebate/blob/master/ci/.travis_test.sh#L16)

## Updating the documentation

We use [mkdocs](http://www.mkdocs.org/) to manage our [documentation](http://EpistasisLab.github.io/scikit-rebate/). This allows us to write the docs in Markdown and compile them to HTML as needed. Below are a few useful commands to know when updating the documentation. Make sure that you are running them in the base repository directory.

* `mkdocs serve`: Hosts of a local version of the documentation that you can access at the provided URL. The local version will update automatically as you save changes to the documentation.

* `mkdocs build --clean`: Creates a fresh build of the documentation in HTML. Always run this before deploying the documentation to GitHub.

* `mkdocs gh-deploy`: Deploys the documentation to GitHub. If you're deploying on your fork of scikit-rebate, the online documentation should be accessible at `http://<YOUR GITHUB USERNAME>.github.io/scikit-rebate/`. Generally, you shouldn't need to run this command because you can view your changes with `mkdocs serve`.

## After submitting your pull request

After submitting your pull request, [Travis-CI](https://travis-ci.com/) will automatically run unit tests on your changes and make sure that your updated code builds and runs on Python 2 and 3. We also use services that automatically check code quality and test coverage.

Check back shortly after submitting your pull request to make sure that your code passes these checks. If any of the checks come back with a red X, then do your best to address the errors.
