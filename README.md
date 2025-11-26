# DeepFake Detection System

This repo is a comprehensive notebook for building a DeepFake Detection System using Deep Learning approaches.

## Project Setup

1. Clone the repository: `git clone https://github.com/DangCongKhai/Mini_GPT_From_Scratch.git`
2. Dependencies: 
- Install uv from [here](https://docs.astral.sh/uv/installation/)
- Install the dependencies: In your terminal, run `uv sync`


## Create a Pull Request (PR)

Follow these steps to create a clear, reviewable pull request:

1. Update your local `develop` branch:

```
git checkout develop
git pull origin develop
```

2. Create a new feature branch (use a descriptive name):

```
git checkout -b feature/short-description
```

Follow git convention here: [Link](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13)

3. Make your changes, run tests/lint locally, then commit:

```
git add .
git commit -m "Short: describe what and why"
```

4. Push the branch to the remote and open a PR:

```
git push -u origin feature/short-description
# Then open a PR on GitHub from your branch into `main`
```

You can also create a PR from the command line with [GitHub CLI](https://cli.github.com):

```
gh pr create --base main --head your-branch --title "Short title" --body "Detailed description"
```

PR checklist (suggested):
- Include a descriptive title and detailed description
- Link related issue(s) if present
- Request reviewers and set the correct target branch

