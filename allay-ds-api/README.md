install virtual environment (in repo root)
```
pipenv install --python=3.7.6
```

enter virtual environment (in repo root)
```
pipenv shell
```

launch locally
```
cd allay-ds-api
uvicorn fastapi_app:APP --reload
```

push to heroku (in repo root)
```
git checkout feature/<your-feature-name>
git subtree push --prefix allay-ds-api staging master
```
