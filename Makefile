install:
	sudo python3 setup.py install

clean:
	sudo rm -rf build/ dist/ *.egg-info __pycache__ */__pycache__ models/ visuals/

commit:
	git commit -a

flake:
	flake8 *.py
	flake8 ac_gym/*.py
	flake8 ac_gym/ptan/actions.py
	flake8 ac_gym/ptan/agent.py
	flake8 ac_gym/ptan/experience.py
	flake8 ac_gym/ptan/ignite.py
	flake8 ac_gym/ptan/common/runfile.py
	flake8 ac_gym/ptan/common/utils.py
	flake8 ac_gym/ptan/common/wrappers.py
	flake8 ac_gym/ptan/common/wrappers_simple.py

