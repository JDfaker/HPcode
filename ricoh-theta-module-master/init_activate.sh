
CUR="$PWD"

if [ ! -d venv ]; then
    if [ ! -d repos-3rd ]; then
        mkdir repos-3rd
    fi
    mkdir venv
    cd repos-3rd/
    git clone https://github.com/pypa/virtualenv
    cd virtualenv
    ./virtualenv.py --python /usr/bin/python2.7 --no-site-packages ../../venv

    cd "$CUR"
fi

echo "alias theta_venv='source $CUR/venv/bin/activate'" >> ~/.bashrc
source ~/.bashrc

theta_venv
