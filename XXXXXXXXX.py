import yaml

from zigzag.parser.WorkloadValidator import WorkloadValidator

with open("inputs/workload/resnet18.yaml", encoding="utf-8") as f:
    data = yaml.safe_load(f)

validator = WorkloadValidator(data)
normalized_data = validator.normalized_data


validate_succes = validator.validate()
if not validate_succes:
    raise ValueError("Failed to validate user provided accelerator.")
