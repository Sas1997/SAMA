# Contributing to SAMA

Thank you for your interest in SAMA. Bug reports, questions, feature suggestions and code contributions are all welcome.

## Reporting bugs

Open an issue at https://github.com/Sas1997/SAMA/issues and include:

- The SAMA version (`pip show samapy`) or the Windows installer version
- Your operating system and Python version
- The configuration file or input data needed to reproduce the problem
- What you expected to happen and what happened instead, including any error message

## Asking for help

Open an issue with the `question` label. Please check the [documentation](./Docs/SAMA_Documentation.md) first, since many parameters and outputs are explained there.

## Suggesting features

Open an issue describing the use case, the component or tariff structure involved, and any references that describe the model you would like to see.

## Contributing code

1. Fork the repository and create a branch from `main`.
2. Install the package in editable mode: `pip install -e .`
3. Make your change. Keep each pull request focused on one fix or feature.
4. Install pytest (`pip install pytest`) and run the test suite with `pytest`. It takes about a minute and must pass before a pull request is merged. If your change affects energy management, results or the optimizers, also run the slow tests with `pytest -m slow` (a few minutes).
5. Open a pull request that explains what changed and why, and link the related issue. The test suite runs automatically on every pull request.

Changes to energy management, component models or economic calculations should explain the physical or financial reasoning and, where possible, include a small example showing the result before and after the change.

## License

By contributing, you agree that your contributions are released under the [GNU Affero General Public License v3.0](./LICENSE).
