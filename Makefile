install:
	sudo python3 setup.py install

clean:
	sudo rm -rf build/ dist/ *.egg-info __pycache__ */__pycache__ models/ visuals/

commit:
	git commit -a

flake:
	flake8 *.py
	flake8 ac_gym/*.py
	flake8 ac_gym/ptan/*.py
	flake8 ac_gym/ptan/common/*.py

