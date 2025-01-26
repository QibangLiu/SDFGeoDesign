# %%
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Model training arguments")
  parser.add_argument(
      "--model", type=str, default="forward")
  print("Hello, World!")
  args, unknown = parser.parse_known_args()

  print(args, unknown)
