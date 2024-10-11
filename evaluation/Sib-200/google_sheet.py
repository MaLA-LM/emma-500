import gspread
from oauth2client.service_account import ServiceAccountCredentials


def write2sheet(sheet_name, result_df):
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        "../gckey.json",
        [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    client = gspread.authorize(credentials)

    try:
        spreadsheet = client.open("MaLA evaluation")
    except gspread.SpreadsheetNotFound:
        spreadsheet = client.create("MaLA evaluation")

    sheet_titles = [sheet.title for sheet in spreadsheet.worksheets()]
    if sheet_name in sheet_titles:
        sheet = spreadsheet.worksheet(sheet_name)
    else:
        sheet = spreadsheet.add_worksheet(title=sheet_name, rows="2000", cols="40")

    existing_data = sheet.get_all_values()
    existing_cols = len(existing_data[0]) if existing_data else 0

    if existing_cols < 2:
        data_list = [result_df.columns.values.tolist()] + result_df.values.tolist()
        sheet.update(range_name="A1", values=data_list)
    else:
        column_letter = gspread.utils.rowcol_to_a1(1, existing_cols + 1)[:-1]
        # print(column_letter)
        data_list = [[result_df.columns[1]]] + [[item] for item in result_df.iloc[:, 1]]
        # print(data_list)
        sheet.update(range_name=f"{column_letter}1", values=data_list)


# data = {
#     'Language': ['English', 'Spanish', 'French'],
#     'Model_2': [0.5, 0.89, 0.92]
# }
# df = pd.DataFrame(data)
# sheet_name = 'SIB-200'
# write2sheet(sheet_name, df)
