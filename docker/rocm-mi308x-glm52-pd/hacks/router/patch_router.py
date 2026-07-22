import sys

filepath = '/sgl-workspace/openai-protocol-patched/src/responses.rs'
with open(filepath, 'r') as f:
    content = f.read()

if 'deserialize_tool_choice' in content:
    print('Already patched')
    sys.exit(0)

old_imports = 'use crate::{builders::ResponsesResponseBuilder, validated::Normalizable};'
new_imports = old_imports + '''

/// Custom deserializer for tool_choice that accepts both string and dict format.
pub fn deserialize_tool_choice<'de, D>(deserializer: D) -> Result<Option<ToolChoice>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: Option<Value> = Option::deserialize(deserializer)?;
    match value {
        None => Ok(None),
        Some(v) => {
            if let Some(obj) = v.as_object() {
                if let Some(type_val) = obj.get("type") {
                    if let Some(type_str) = type_val.as_str() {
                        match type_str {
                            "auto" => return Ok(Some(ToolChoice::Value(ToolChoiceValue::Auto))),
                            "required" => return Ok(Some(ToolChoice::Value(ToolChoiceValue::Required))),
                            "none" => return Ok(Some(ToolChoice::Value(ToolChoiceValue::None))),
                            _ => {}
                        }
                    }
                }
            }
            Ok(Some(serde_json::from_value(v).map_err(serde::de::Error::custom)?))
        }
    }
}'''

content = content.replace(old_imports, new_imports, 1)

old_field = '''    /// Tool choice behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,'''

new_field = '''    /// Tool choice behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(deserialize_with = "deserialize_tool_choice")]
    pub tool_choice: Option<ToolChoice>,'''

content = content.replace(old_field, new_field, 1)

with open(filepath, 'w') as f:
    f.write(content)

print('Patched responses.rs successfully')
