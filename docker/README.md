# Build plan

1. Make htm.core image
   1. from slim
   2. apt-get update/install && rm
   3. set python venv
   4. clone
   5. pip install && rm
2. Make hima image
   1. preinstall common requirements
      1. from slim
      2. apt-get update/install minimal && rm
      3. install oh-my-zsh
   2. get CoppeliaSim to `/app/deps/
      1. set env vars
      2. wget && extract && rm xz
   3. install pre-reqs for PyRep
      1. apt-get update/install qt and etc && rm
   4. ===== start building python venv ====
   5. copy htm.core to `/app/deps/htm.core`
      1. copy venv from htm.core
   6. install pyrep to `/app/deps/PyRep`
      1. git clone && pip install req && pip install pyrep && rm PyRep
   7. install hima
      1. get hima requirements to `/app/hima`
      2. pip install reqs + pytorch
      3. git clone hima && pip install -e .
3. Mount dev hima
   1. from hima image
   2. rm `/app/hima`
   3. bind local hima to `/app/hima`

## Folder structure

`/app`:

- `/app/hima` - hima is cloned here (I renamed it from `htm-rl`)  
- `/app/deps` - root for dependecies. PyRep and CoppeliaSim live here. Htm.core is also built from here, although, I remove it afterwards.

## Lesser docker image

- `pip install --no-cache-dir` prevents storing intermediate installation artifacts
- `&& rm -rf /var/lib/apt/lists/*` after `apt-get ...` removes unnescessary artifacts
- `git clone --depth 1 --branch master` clone only the last commit
- `rm ...` remove archives or any other not required files after you used them, e.g. CoppeliaSim archive or PyRep sources.

## Other notes

- `DEBIAN_FRONTEND=noninteractive apt-get install -y xxxx` prevents confirmation prompts or answer yes. Otherwise it can hang build process!
