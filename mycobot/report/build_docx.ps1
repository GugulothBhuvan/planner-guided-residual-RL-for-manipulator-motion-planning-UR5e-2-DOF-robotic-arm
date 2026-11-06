# Build report.docx from report.html (Windows, Microsoft Word installed):
#   1. python html_to_docx.py          (needs python-docx, beautifulsoup4, qrcode[pil]; run with the project or a scratch venv)
#   2. this script: Word updates the contents field and the page numbers and saves the file.
# Usage: powershell -File build_docx.ps1
$d = Split-Path -Parent $MyInvocation.MyCommand.Path
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open("$d\report.docx", $false, $false)
    $doc.Repaginate()
    $doc.Fields.Update() | Out-Null
    foreach ($t in $doc.TablesOfContents) { $t.Update() }
    "pages: " + $doc.ComputeStatistics(2)
    $doc.Save()
    $doc.Close($false)
} finally { $word.Quit() }
