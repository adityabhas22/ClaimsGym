from claimsops_env.server import app, create_app, main as _main

__all__ = ["app", "create_app", "main"]


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
