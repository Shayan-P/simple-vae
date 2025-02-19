import src
import code
import pkgutil
import sys

def invalidate_cache():
    for module_info in pkgutil.walk_packages(src.__path__, src.__name__ + "."):
        if module_info.name in sys.modules:
            del sys.modules[module_info.name]

def gmm_main():
    invalidate_cache()
    import src.train
    src.train.train_gmm_main()

def vae_main():
    invalidate_cache()
    import src.train
    src.train.train_vae_main()

if __name__ == "__main__":
    funcs = {
        name: func
        for name, func in globals().items()
        if not name.startswith("_") and name != "invalidate_cache" and name != "main" and callable(func)
    }
    if len(sys.argv) > 2:
        print(f"Usage: python run.py [{', '.join(funcs.keys())}]")
        exit(1)
    elif len(sys.argv) == 2:
        name = sys.argv[1]
        if name not in funcs:
            print(f"Usage: python run.py [{', '.join(funcs.keys())}]")
            exit(1)
        funcs[name]()
    else:
        code.interact(local=locals())
