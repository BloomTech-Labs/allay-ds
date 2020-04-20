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
git subtree push --prefix allay-ds-api heroku master
```
