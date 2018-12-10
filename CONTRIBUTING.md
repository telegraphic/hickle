## Contributing to hickle

Thanks for thinking about contributing to hickle, improvements and bugfixes are most welcome. 

The following is a brief set of guidelines (not rules) for contributing:

* **Be nice.** Please follow the [code of conduct](https://github.com/telegraphic/hickle/blob/master/CODE_OF_CONDUCT.md).
* **Squashing bugs.** If you find a bug, please [open an issue](https://github.com/telegraphic/hickle/issues), with some simple steps on how to reproduce it. Try not to make duplicate requests.
* **Feature requests.** Feel free to make feature requests, also by [opening an issue](https://github.com/telegraphic/hickle/issues). Be clear about what it is and why it would be awesome.
* **Pull requests.** If you add a cool feature you think would be useful broadly, please issue a pull request with some notes on what it does.
* **Git comments.** Try and make these clear, even if concise. 
* **Major changes.** As quite a few people use this package, we have tried to maintain backwards compatibility as much as possible. As such, please open a discussion before you start your quest, to make sure the changes can be merged without upset.
* **Unit tests.** If you add new functionality, please write a unit test (if you're familiar with how to). This should be places in the `./tests` directory, and will run with py.test.
* **Travis-CI.** When you issue a pull request, Travis-CI will automatically run the unit tests. You can test yourself by running `cd tests; coverage run --source=hickle -m py.test`.
* **Style.** Try and keep your code Py2.7 and Py3 compatible, and roughly follow [PEP8](https://www.python.org/dev/peps/pep-0008/) with [google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
* **Beginners welcome.** If you're not yet comfortable with some of the fancier things mentioned above, do your best! Just package up your idea/thought/code and [open an issue](https://github.com/telegraphic/hickle/issues) with some clear details.

That's about it. Happy contributing!
