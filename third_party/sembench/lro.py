def run_match113():
    sp = get_spark()
    home = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/"
             "home_office_senior_officials_travel_data_return.csv")
    )
    left_df = (
        home.select(col("Name of Official").alias("left_name"))
            .dropna().dropDuplicates(["left_name"])
    )

    travel = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/"
             "travel-exp-April-June-2018.csv")
    )
    right_df = (
        travel.select(col("Senior Officials Name").alias("right_name"))
              .dropna().dropDuplicates(["right_name"])
    )

    prompt = """Determine whether the two senior government official names share the same first name (first given name, ignoring middle names / hyphenated parts).

Official A: {left_name}
Official B: {right_name}"""

    truth = [
        ('Mark Sedwill ',  'Mark Bryson-Richardson'),
        ('Richard Clarke', 'Richard Montgomery'),
    ]
    gt_path = _write_match_gt_csv("match113", truth)
    _run_match_join("match113", left_df, right_df, prompt, gt_path,
                    "left_name", "right_name")


def run_match121():
    sp = get_spark()
    april = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/01.Apr_2018.csv")
    )
    left_df = (
        april.select(col("Supplier").alias("left_Supplier"))
             .dropna().dropDuplicates(["left_Supplier"])
    )

    may = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/santos/2015_05_expenditure.csv")
    )
    right_df = (
        may.select(col("Supplier").alias("right_Supplier"))
           .dropna().dropDuplicates(["right_Supplier"])
    )

    prompt = """Determine whether these two supplier names refer to the same organisation (ignoring case, abbreviations, punctuation, trailing legal suffixes such as Ltd/Limited, and minor formatting differences).

Supplier A: {left_Supplier}
Supplier B: {right_Supplier}"""

    truth = [
        ('Nhs Supply Chain',                 'NHS SUPPLY CHAIN'),
        ('Novartis Pharmaceuticals Uk Ltd',  'NOVARTIS PHARMACEUTICALS UK LTD'),
        ('Roche Diagnostics Limited',        'ROCHE PRODUCTS LTD'),
        ('Csc',                              'CSC COMPUTER SCIENCES LTD'),
        ('Philips Healthcare',               'PHILIPS HEALTHCARE'),
        ('Nhs Blood And Transplant',         'NHS BLOOD & TRANSPLANT'),
        ('NHS Litigation Authority',         'NHS LITIGATION AUTHORITY'),
    ]
    gt_path = _write_match_gt_csv("match121", truth)
    _run_match_join("match121", left_df, right_df, prompt, gt_path,
                    "left_Supplier", "right_Supplier")


def run_match202():
    sp = get_spark()
    yelp = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/restaurants2/yelp.csv")
    )
    yelp = yelp.filter(col("zip") == 60642)
    left_df = yelp.select(
        col("ID").cast("int").alias("left_ID"),
        col("name").alias("left_name"),
        col("address").alias("left_address"),
        col("phone").alias("left_phone"),
        col("cuisine").alias("left_cuisine"),
    )

    zomato = (
        sp.read.option("header", "true").option("multiLine", "true")
        .option("escape", '"').option("inferSchema", "true")
        .csv(f"{LROBENCH_DATABASES_DIR}/restaurants2/zomato.csv")
    )
    zomato = zomato.filter(col("zip") == 60642)
    right_df = zomato.select(
        col("ID").cast("int").alias("right_ID"),
        col("name").alias("right_name"),
        col("address").alias("right_address"),
        col("phone").alias("right_phone"),
        col("cuisine").alias("right_cuisine"),
    )

    prompt = """Determine whether the two restaurant records refer to the same real-world restaurant (same establishment at the same address).

Yelp: name={left_name} | address={left_address} | phone={left_phone} | cuisine={left_cuisine}
Zomato: name={right_name} | address={right_address} | phone={right_phone} | cuisine={right_cuisine}"""

    truth = [
        (2, 844), (72, 1020), (90, 744), (111, 564),
        (198, 1022), (275, 272), (333, 134), (418, 590),
    ]
    gt_path = _write_match_gt_csv("match202", truth)
    _run_match_join("match202", left_df, right_df, prompt, gt_path,
                    "left_ID", "right_ID")


def run_match305():
    sp = get_spark()
    left_df = _column_descriptor_df(
        sp, f"{LROBENCH_DATABASES_DIR}/california_schools/frpm.csv", "left"
    )
    right_df = _column_descriptor_df(
        sp, f"{LROBENCH_DATABASES_DIR}/california_schools/schools.csv", "right"
    )

    prompt = """Determine whether the two columns from different tables describe the same real-world feature (same semantic attribute), using the column names and sample values.

Table A column: name={left_colname} | samples={left_samples}
Table B column: name={right_colname} | samples={right_samples}"""

    truth = [
        ('CDSCode',                  'CDSCode'),
        ('County Code',              'County'),
        ('District Code',            'NCESDist'),
        ('District Name',            'District'),
        ('School Name',              'School'),
        ('District Type ',           'DOCType'),
        ('School Type',              'SOCType'),
        ('Educational Option Type',  'EdOpsName'),
        ('Charter School (Y/N)',     'Charter'),
        ('Charter School Number',    'CharterNum'),
        ('Charter Funding Type',     'FundingType'),
    ]
    gt_path = _write_match_gt_csv("match305", truth)
    _run_match_join("match305", left_df, right_df, prompt, gt_path,
                    "left_colname", "right_colname")

