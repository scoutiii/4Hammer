const vscode = require("vscode");

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function lineIndent(line) {
  const match = line.match(/^\s*/);
  return match ? match[0].length : 0;
}

function isEnumMemberCandidate(trimmed) {
  if (trimmed.startsWith("#")) {
    return false;
  }
  if (/^(fun|act|cls|using|enum)\b/.test(trimmed)) {
    return false;
  }
  return /^[A-Za-z_][A-Za-z0-9_]*\b/.test(trimmed);
}

function findQualifier(lineText, wordRange) {
  const prefix = lineText.slice(0, wordRange.start.character);
  const match = prefix.match(/([A-Za-z_][A-Za-z0-9_]*)\s*::\s*$/);
  return match ? match[1] : null;
}

async function findDefinitions(word, qualifier) {
  const files = await vscode.workspace.findFiles("**/*.rl", "**/node_modules/**");
  const qualifiedResults = [];
  const generalResults = [];
  const wordPattern = new RegExp(`\\b${escapeRegex(word)}\\b`);
  const funPattern = new RegExp(
    `^\\s*fun(?:<[^>]+>)?\\s+${escapeRegex(word)}\\b`
  );
  const actPattern = new RegExp(`^\\s*act\\s+${escapeRegex(word)}\\b`);
  const enumPattern = new RegExp(`^\\s*enum\\s+${escapeRegex(word)}\\b`);
  const clsPattern = new RegExp(`^\\s*cls\\s+${escapeRegex(word)}\\b`);
  const usingPattern = new RegExp(`^\\s*using\\s+${escapeRegex(word)}\\b`);
  const enumHeaderPattern = /^\s*enum\s+([A-Za-z_][A-Za-z0-9_]*)\b/;

  for (const uri of files) {
    const document = await vscode.workspace.openTextDocument(uri);
    const lines = document.getText().split(/\r?\n/);
    let enumContext = null;

    for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
      const line = lines[lineIndex];
      const trimmed = line.trim();
      const indent = lineIndent(line);

      if (
        enumContext &&
        trimmed !== "" &&
        !trimmed.startsWith("#") &&
        indent <= enumContext.indent
      ) {
        enumContext = null;
      }

      if (!trimmed.startsWith("#")) {
        const enumMatch = line.match(enumHeaderPattern);
        if (enumMatch) {
          enumContext = {
            name: enumMatch[1],
            indent,
            memberIndent: null,
          };
          if (enumPattern.test(line)) {
            const column = line.search(wordPattern);
            const position = new vscode.Position(lineIndex, Math.max(0, column));
            generalResults.push(new vscode.Location(uri, position));
          }
        }
      }

      if (
        funPattern.test(line) ||
        actPattern.test(line) ||
        clsPattern.test(line) ||
        usingPattern.test(line)
      ) {
        const column = line.search(wordPattern);
        const position = new vscode.Position(lineIndex, Math.max(0, column));
        generalResults.push(new vscode.Location(uri, position));
      }

      if (!enumContext) {
        continue;
      }

      if (trimmed === "" || trimmed.startsWith("#")) {
        continue;
      }

      if (enumContext.memberIndent === null) {
        if (isEnumMemberCandidate(trimmed)) {
          enumContext.memberIndent = indent;
        }
      }

      if (
        enumContext.memberIndent !== null &&
        indent === enumContext.memberIndent &&
        isEnumMemberCandidate(trimmed)
      ) {
        const memberMatch = trimmed.match(
          /^([A-Za-z_][A-Za-z0-9_]*)(?::|\b)/
        );
        if (!memberMatch || memberMatch[1] !== word) {
          continue;
        }
        const column = line.search(wordPattern);
        const position = new vscode.Position(lineIndex, Math.max(0, column));
        const location = new vscode.Location(uri, position);
        if (qualifier && enumContext.name === qualifier) {
          qualifiedResults.push(location);
        } else {
          generalResults.push(location);
        }
      }
    }
  }

  return qualifiedResults.length > 0 ? qualifiedResults : generalResults;
}

function activate(context) {
  const provider = {
    async provideDefinition(document, position) {
      const wordRange = document.getWordRangeAtPosition(
        position,
        /[A-Za-z_][A-Za-z0-9_]*/
      );
      if (!wordRange) {
        return null;
      }

      const word = document.getText(wordRange);
      if (!word) {
        return null;
      }

      const lineText = document.lineAt(position.line).text;
      const qualifier = findQualifier(lineText, wordRange);
      const locations = await findDefinitions(word, qualifier);
      if (locations.length === 0) {
        return null;
      }

      return locations;
    },
  };

  context.subscriptions.push(
    vscode.languages.registerDefinitionProvider(
      { language: "rlc", scheme: "file" },
      provider
    )
  );
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
