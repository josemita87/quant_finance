function generateFormFromSheet() {
  // 1. SETUP
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var data = sheet.getDataRange().getValues();

  // Remove the header row
  var headers = data.shift();

  // Create the new Form
  var formTitle = "ECRAN Sunscreen Survey (" + new Date().toLocaleDateString() + ")";
  var form = FormApp.create(formTitle);

  form.setDescription("We are conducting a study on sunscreen usage habits. Your responses will help us improve our products. This survey takes about 5 minutes.");

  // 2. LOOP THROUGH DATA ROWS
  // Column Mapping: 0=Question Title, 1=Question Type, 2=Options, 3=Required

  data.forEach(function (row) {
    var title = row[0];
    var type = row[1].toString().toUpperCase().trim();
    var optionsRaw = row[2];
    var isRequired = row[3];

    // Skip empty rows
    if (!title) return;

    var item;

    // 3. SWITCH BASED ON QUESTION TYPE
    switch (type) {
      case "TEXT":
        item = form.addTextItem();
        item.setTitle(title);
        break;

      case "PARAGRAPH":
        item = form.addParagraphTextItem();
        item.setTitle(title);
        break;

      case "MULTIPLE_CHOICE":
        item = form.addMultipleChoiceItem();
        item.setTitle(title);
        var choices = optionsRaw.toString().split(",").map(function (o) { return o.trim() });
        // Filter empty choices
        choices = choices.filter(function (c) { return c !== ""; });
        item.setChoiceValues(choices);
        break;

      case "CHECKBOX":
        item = form.addCheckboxItem();
        item.setTitle(title);
        var choices = optionsRaw.toString().split(",").map(function (o) { return o.trim() });
        choices = choices.filter(function (c) { return c !== ""; });
        item.setChoiceValues(choices);
        break;

      case "DROPDOWN":
        item = form.addListItem();
        item.setTitle(title);
        var choices = optionsRaw.toString().split(",").map(function (o) { return o.trim() });
        choices = choices.filter(function (c) { return c !== ""; });
        item.setChoiceValues(choices);
        break;

      case "DATE":
        item = form.addDateItem();
        item.setTitle(title);
        break;

      case "SCALE":
        item = form.addScaleItem();
        item.setTitle(title);
        // Assumes options are "Min, Max" (e.g., "1, 10")
        var scaleParts = optionsRaw.toString().split(",");
        if (scaleParts.length >= 2) {
          var min = parseInt(scaleParts[0]);
          var max = parseInt(scaleParts[1]);
          // Google Forms Scale limits: 0 or 1 to 10
          if (min < 0) min = 0;
          if (max > 10) max = 10;
          item.setBounds(min, max);

          // Optional: Set labels for endpoints if provided in options?
          // For now, simple numeric bounds.
        } else {
          item.setBounds(1, 5); // Default fallback
        }
        break;

      default:
        // Default to short text if type is unknown
        item = form.addTextItem();
        item.setTitle(title + " (Type Unrecognized)");
    }

    // Set Required Status
    // Check for boolean true or string "TRUE" (case insensitive)
    var req = false;
    if (isRequired === true || isRequired.toString().toUpperCase() === "TRUE") {
      req = true;
    }

    if (item) {
      item.setRequired(req);
    }
  });

  // 4. OUTPUT LINK
  Logger.log("Form created! Edit URL: " + form.getEditUrl());
  Logger.log("Published URL: " + form.getPublishedUrl());
  SpreadsheetApp.getUi().alert("Form Created Successfully! \n\nEdit Link: " + form.getEditUrl() + "\n\nPublished Link: " + form.getPublishedUrl());
}

/**
 * Uploads responses from the active sheet to the Google Form.
 * Requires the Form Edit URL to be provided.
 */
function uploadResponses() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var data = sheet.getDataRange().getValues();
  var headers = data.shift(); // Remove header row

  // PROMPT USER FOR FORM URL
  var ui = SpreadsheetApp.getUi();
  var response = ui.prompt('Enter Form Edit URL', 'Please paste the URL of the Google Form you want to populate:', ui.ButtonSet.OK_CANCEL);

  if (response.getSelectedButton() !== ui.Button.OK) {
    return;
  }

  var formUrl = response.getResponseText();
  if (!formUrl) {
    ui.alert('URL cannot be empty.');
    return;
  }

  try {
    var form = FormApp.openByUrl(formUrl);
  } catch (e) {
    ui.alert('Error opening Form. Please check the URL. Error: ' + e.message);
    return;
  }

  var items = form.getItems();
  var itemMap = {};

  // Map Question Titles to Item Objects
  items.forEach(function (item) {
    itemMap[item.getTitle()] = item;
  });

  var successCount = 0;
  var errorCount = 0;

  // Iterate through rows
  data.forEach(function (row) {
    var formResponse = form.createResponse();
    var hasData = false;

    row.forEach(function (cellValue, index) {
      var questionTitle = headers[index];
      var item = itemMap[questionTitle];

      if (item && cellValue !== "") {
        var responseItem;

        // Handle different item types
        switch (item.getType()) {
          case FormApp.ItemType.TEXT:
            responseItem = item.asTextItem().createResponse(cellValue.toString());
            break;
          case FormApp.ItemType.PARAGRAPH_TEXT:
            responseItem = item.asParagraphTextItem().createResponse(cellValue.toString());
            break;
          case FormApp.ItemType.MULTIPLE_CHOICE:
            responseItem = item.asMultipleChoiceItem().createResponse(cellValue.toString());
            break;
          case FormApp.ItemType.LIST: // Dropdown
            responseItem = item.asListItem().createResponse(cellValue.toString());
            break;
          case FormApp.ItemType.CHECKBOX:
            // Split comma-separated values
            var choices = cellValue.toString().split(",").map(function (s) { return s.trim() });
            responseItem = item.asCheckboxItem().createResponse(choices);
            break;
          case FormApp.ItemType.SCALE:
            // Scale expects an integer
            responseItem = item.asScaleItem().createResponse(parseInt(cellValue));
            break;
          case FormApp.ItemType.DATE:
            // Date expects a Date object or string in YYYY-MM-DD
            responseItem = item.asDateItem().createResponse(new Date(cellValue));
            break;
        }

        if (responseItem) {
          formResponse.withItemResponse(responseItem);
          hasData = true;
        }
      }
    });

    if (hasData) {
      try {
        formResponse.submit();
        successCount++;
      } catch (e) {
        Logger.log("Error submitting row: " + e.message);
        errorCount++;
      }
    }
  });

  ui.alert('Upload Complete!\nSuccessfully submitted: ' + successCount + '\nErrors: ' + errorCount);
}
