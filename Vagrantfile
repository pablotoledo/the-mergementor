# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
    config.vm.box = "ubuntu/jammy64" # Ubuntu 22.04
    config.vm.box_check_update = true
    config.vm.hostname = "MergeMentor" # Agregamos el nombre de la maquina virtual
    config.vm.network "private_network", ip: "192.168.33.10"

  
    config.vm.provider "virtualbox" do |vb|
        vb.memory = "16384"
        vb.cpus = 4
        vb.gui = true
    end

    # Set password for vagrant user
    config.vm.provision "shell", inline: <<-SHELL
        # Set password for vagrant user
        echo 'vagrant:vagrant' | chpasswd
    SHELL
  
    # Setting SSH Server
    config.vm.provision "shell", inline: <<-SHELL
        sudo apt-get install -y openssh-server
        sudo systemctl enable ssh
        sudo systemctl start ssh
        sudo sed -i 's/^#\\?PasswordAuthentication.*$/PasswordAuthentication yes/' /etc/ssh/sshd_config
        sudo systemctl restart ssh
    SHELL
  
    config.vm.provision "shell", inline: <<-SHELL
        apt-get update
        apt-get upgrade -y
        apt-get install -y python3 python3-pip
        pip3 install torch torchvision -f https://download.pytorch.org/whl/cpu/torch_stable.html
        pip3 install transformers requests sympy gradio
    SHELL

    config.vm.synced_folder ".", "/vagrant_data"
  end
