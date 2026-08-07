import great_expectations as gx 
import pandas as pd 
import yaml
import os

def load_contract(path = "data/contract.yaml"):
    """ Load data contract from YAML file """
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Data contract not found at {path}")
        print("Define Data Contract in Yaml file first")
        return None 

def generate_expectations_from_contract(contract):
    """ Generate gx expectations from contract definition. This is generic- it 
    works for ANY file defined in the contract """

    expectations_list = []

    #1 Row Count Check
    if 'row_count_min' in contract and 'row_count_max' in contract:
        expectations_list.append(
            gx.expectations.ExpectTableRowCountToBeBetween(
            min_value= contract['row_count_min'],
            max_value= contract['row_count_max']
            )
        )

    #2 Column Checks
    for col_name, col_rules in contract["columns"].items():
       
        #Nullable
        if not col_rules.get('nullable',True):
            expectations_list.append(
                gx.expectations.ExpectColumnValuesToNotBeNull(column=col_name)
            )

        #Unique
        if col_rules.get('unique',False):
            expectations_list.append(
                gx.expectations.ExpectColumnValuesToBeUnique(column=col_name)
            )

        #Values (Set Membership)
        if 'allowed_values' in col_rules:
            expectations_list.append(
                gx.expectations.ExpectColumnValuesToBeInSet(
                    column=col_name,
                    value_set=col_rules['allowed_values']
                )
            )

        #Min/Max Values
        mostly = col_rules.get('mostly',1.0)
        if 'min' in col_rules and 'max' in col_rules:
            expectations_list.append(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=col_name,
                    min_value=col_rules['min'],
                    max_value=col_rules['max'],
                    mostly=mostly
                )
            )
        elif 'max' in col_rules:
            expectations_list.append(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=col_name,
                    min_value=None,
                    max_value=col_rules['max'],
                    mostly=mostly
                )
            )
        elif 'min' in col_rules:
            expectations_list.append(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=col_name,
                    min_value=col_rules['min'],
                    max_value=None,
                    mostly=mostly
                )
            )    
        #Type Validation is handled by pandas dtype not great expectations.
        #We will check it seperately 
    return expectations_list

def apply_sentinels(df, contract_columns):
    """Replace any contract-declared sentinel_values with NaN, on a copy of df
    . USed only for GX. So Placeholder codes like 365243 aren't scored as out of range
    or as null-check failures"""
    sentinel_cols = {
    col_name: col_rules['sentinel_values']
    for col_name, col_rules in contract_columns.items()
    if col_rules.get('sentinel_values') and col_name in df.columns
    }

    if not sentinel_cols:
        return df  # nothing to do — skip the copy entirely

    df_clean = df.copy()
    for col_name, sentinels in sentinel_cols.items():
        mask = df_clean[col_name].isin(sentinels)
        df_clean.loc[mask, col_name] = pd.NA

    return df_clean

def validate_file_with_contract(file_name, file_path, contract):
    """Validate file against its contract. 
    Returns [Passed/List of Errors]"""

    print("\n"+"=" * 60)
    print(f"VALIDATING......{file_name}")
    print("=" * 60)

    #1 Check file path
    if not os.path.exists(file_path):
        print(f"{file_name} does not exist at {file_path}")
        return False, ["File Not Found"]

    #Load file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"failed to read: {file_path}")
        print(f"Error: {e}")
        return False, [f"Failed to read csv: {e}"]
    print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    #3 validate column name against contract
    actual_cols = set(df.columns)
    expected_cols = set(contract['columns'].keys())

    errors = []

    if expected_cols - actual_cols:
        missing = expected_cols - actual_cols
        errors.append(f"Missing Columns: {missing}")
    
    #4 Check Data Type
    TYPE_MAP = {
    "integer": {"int8", "int16", "int32", "int64", "Int64"},
    "float": {"float16", "float32", "float64"},
    "string": {"object", "string", "str"}
    }

    for col_name, col_rules in contract['columns'].items():
        if col_name in df.columns:
            expected_type = col_rules.get('type')
            if expected_type:
                actual_type = str(df[col_name].dtype)
                allowed_types = TYPE_MAP.get(expected_type)
                if not allowed_types:
                    print(f"{col_name} has unidentified data_type: {expected_type}")
                elif actual_type not in allowed_types:
                    errors.append(f"Col '{col_name}' expected type {expected_type}, got {actual_type}")

    #5 Run gx Validation for value-level checks
    df_values = apply_sentinels(df, contract['columns'])

    context = gx.get_context(mode='ephemeral') 
    data_source = context.data_sources.add_pandas(f"pandas_{file_name}")
    data_asset = data_source.add_dataframe_asset(name=file_name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe("whole_dataset")
    batch = batch_definition.get_batch(batch_parameters={"dataframe":df_values})
    
    expectations = generate_expectations_from_contract(contract)

    print(f"\n Running {len(expectations)} Great Expectations checks")

    passed_check = 0
    failed_check = 0

    for exp in expectations:
        result = batch.validate(exp)
        if result.success:
            passed_check += 1
        else:
            failed_check += 1
            try:
                detail = result.result
                errors.append(f"{exp.__class__.__name__} : {detail}")
            except Exception:
                errors.append(f"{exp.__class__.__name__} : Failed")
    print(f"Passed Checks: {passed_check}, failed Checks: {failed_check}")

    if errors:
        print(f"\n {file_name}: failed with {len(errors)} tests")
        return False, errors
    else:
        print(f"\n {file_name}: PASSED ALL  VALIDATION TESTS")
        return True, []

def validate_all_raw_data():
    """Validate all raw csvs against their contracts"""
    print("\n"+"=" * 60)
    print("DATA VALIDATION AGAINST CONTRACT")
    print("=" * 60)

    #1 Load the contract
    contract = load_contract()
    if contract is None:
        print("Contract not found while running validate_all_raw_data")
        return False

    #2 Validate each file
    all_passed = True
    all_errors = {}

    #3 Running contract against each file
    for file_name, file_contract in contract.items():
        file_path = file_contract['path']
        passed, errors = validate_file_with_contract(file_name, file_path, file_contract)
        if not passed:
            all_passed = False
        all_errors[file_name] = errors

    #4 Final Summary of Validation
    print(f"\n"+"=" * 60)
    print("FINAL SUMMARY")
    print(f"=" * 60)

    for file_name, errors in all_errors.items():
        if errors:
            print(f"{file_name}: FAILED")
            for error in errors:
                print(f"    -{error}")
        else:
            print(f"\n{file_name}: PASSED ALL TESTS")
    
    if all_passed:
        print("\n ALL FILES PASSED. PIPELINE CAN PROCEED")
    else:
        print("\n SOME FILES FAILED: PLEASE INVESTIGATE BEFORE PROCEEDING")
        print("\nCommon Fixes:")
        print("1. Update data/contract.yaml to match actual data")
        print("2. Fix Data if Corrupted")

    return all_passed


if __name__ == "__main__":
    validate_all_raw_data()


