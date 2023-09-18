# Mergementor
A way to use https://huggingface.co/microsoft/codereviewer to review your code. And provide you with suggestions to improve your code.


## Creating the Development Environment

This project uses __Vagrant__ as a way to create the development environment. To install Vagrant, follow the instructions on the [Vagrant website](https://www.vagrantup.com/downloads.html).

Ensure you have also installed VirtualBox, which is used by Vagrant to create the virtual machine. To __install VirtualBox__, follow the instructions on the [VirtualBox website](https://www.virtualbox.org/wiki/Downloads). And verify you have installed the __VirtualBox Extension Pack__.

Once you have installed Vagrant and VirtualBox, you can create the development environment by running the following command in the root of the project:

```bash
    vagrant up
```

This will create the virtual machine and install all the dependencies. Once the virtual machine is created, the content of the project will be available in the `/vagrant` folder. In order to access the virtual machine, you can run the following command:

```bash
    vagrant ssh
```

## Prepare the environment

The project needs some information to be configured in your VM, this information is sensitive to be saved in Git. So, please, using the ``_variables.env`` file, collect all information you need an export the environment varibles in your VM.

You can export manualy by running the command on each new terminal you open, or, modify the .bashrc file using ``nano $HOME/.bashrc`` to add the completed content of ``_variables.env`` at the end of the file.

You need to pay attention about where you can find the information that ``_variables.env`` is requiring:

- export GITHUB_USER='': Your GitHub username
- export GITHUB_TOKEN='ghp_...': Your GitHub token, you have to create one in your GitHub account
- export GITHUB_URL='': Your GitHub URL, for example: https://github.com
- export GITHUB_API_URL='': Your GitHub API URL, for example: https://api.github.com

__But if you are using an on-premise GitHub, you need to ask your GitHub administrator to provide you the information about the GitHub URL and GitHub API URL.__


# Running the project

Check the internal TODOs to make some changes in the code before to run the project. This will able you to define a sample case of use to test the project.

To run the project, you need to run the following command:

```bash
    python3 review_onpremise.py
```