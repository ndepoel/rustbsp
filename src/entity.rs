use std::str::Chars;
use std::collections::HashMap;

use super::math;

#[derive(Default, Debug)]
pub struct Entity
{
    pub class_name: String,
    pub origin: math::Vector3,
    pub properties: HashMap<String, String>,
}

fn next_token(chars: &mut Chars<'_>) -> Option<String>
{
    let mut item: Option<char>;

    // Skip whitespace
    loop
    {
        item = chars.next();
        match item
        {
            Some(c) if c.is_whitespace() || c.is_ascii_control() => continue,
            Some(_) => break,
            None => return None,
        }
    }

    let mut token = String::new();

    if item.unwrap() == '\"'
    {
        // Handle quoted strings
        loop
        {
            item = chars.next();

            match item
            {
                Some('\"') => break,
                Some(c) => token.push(c),
                None => break,
            }
        }
    }
    else
    {
        // Handle single word
        loop
        {
            match item
            {
                Some(c) if c.is_whitespace() || c.is_ascii_control() => break,
                Some(c) => token.push(c),
                None => break,
            }

            item = chars.next();
        }
    }

    if token.len() > 0 { Some(token) } else { None }
}

pub fn tokenize(string: &String) -> Vec<String>
{
    let mut tokens = Vec::new();

    let mut chars = string.chars();
    while let Some(token) = next_token(&mut chars)
    {
        tokens.push(token);
    }

    tokens
}

fn parse_vector(string: &String) -> math::Vector3
{
    let mut chars = string.chars();
    math::Vector3
    {
        x: next_token(&mut chars).unwrap_or_default().parse::<f32>().unwrap_or_default(),
        y: next_token(&mut chars).unwrap_or_default().parse::<f32>().unwrap_or_default(),
        z: next_token(&mut chars).unwrap_or_default().parse::<f32>().unwrap_or_default(),
    }
}

fn parse_entity(chars: &mut Chars<'_>) -> Option<Entity>
{
    loop
    {
        match next_token(chars)
        {
            Some(token) if token == "{" => break,
            Some(_) => continue,
            None => return None,
        }
    }

    let mut entity = Entity::default();
    loop
    {
        match next_token(chars)
        {
            Some(token) if token == "}" => break,
            Some(key) => { entity.properties.insert(key, next_token(chars).unwrap_or_default()); },
            None => break,
        }
    }

    if let Some(class_name) = entity.properties.remove("classname")
    {
        entity.class_name = class_name.clone();
    }

    if let Some(origin) = entity.properties.remove("origin")
    {
        entity.origin = parse_vector(&origin);
    }

    Some(entity)
}

pub fn parse_entities(string: &String) -> Vec<Entity>
{
    let mut entities = Vec::new();
    let mut chars = string.chars();

    loop
    {
        match parse_entity(&mut chars)
        {
            Some(entity) => entities.push(entity),
            None => return entities,
        }
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn test_tokenize()
    {
        // Test a mix of quoted strings, spaces, tabs, newlines, longer whitespaces, interior nuls, and trailing whitespace
        let string = "This is \"a test\"   {  \tHello!   \0 \t\r\n   \"I am a longer quoted string\"   \"42 13.0 69\"     }\r\n\0".to_string();
        let mut chars = string.chars();

        assert_eq!(Some("This".to_string()), next_token(&mut chars));
        assert_eq!(Some("is".to_string()), next_token(&mut chars));
        assert_eq!(Some("a test".to_string()), next_token(&mut chars));
        assert_eq!(Some("{".to_string()), next_token(&mut chars));
        assert_eq!(Some("Hello!".to_string()), next_token(&mut chars));
        assert_eq!(Some("I am a longer quoted string".to_string()), next_token(&mut chars));
        assert_eq!(Some("42 13.0 69".to_string()), next_token(&mut chars));
        assert_eq!(Some("}".to_string()), next_token(&mut chars));
        assert_eq!(None, next_token(&mut chars));
    }

    #[test]
    fn test_vector()
    {
        // Test a mix of integer and floating point string values
        let vec = parse_vector(&"1 2.0 3.14".to_string());
        assert_eq!(1.0, vec.x);
        assert_eq!(2.0, vec.y);
        assert_eq!(3.14, vec.z);

        // Missing tokens should result in a default component value
        let vec = parse_vector(&"4 5".to_string());
        assert_eq!(4.0, vec.x);
        assert_eq!(5.0, vec.y);
        assert_eq!(0.0, vec.z);

        // Empty strings should be accepted, result in a default vector
        let vec = parse_vector(&"".to_string());
        assert_eq!(0.0, vec.x);
        assert_eq!(0.0, vec.y);
        assert_eq!(0.0, vec.z);

        // Test parse errors, invalid numbers should result in default values
        let vec = parse_vector(&"a b c".to_string());
        assert_eq!(0.0, vec.x);
        assert_eq!(0.0, vec.y);
        assert_eq!(0.0, vec.z);
    }

    #[test]
    fn test_entity()
    {
        let string = "{ 
            \"classname\" \"foobar\"
            \"origin\" \"1.1 2.2 3.3\"
            \"bar\" \"baz\"
        }".to_string();

        let mut chars = string.chars();
        let ent = parse_entity(&mut chars);
        assert!(ent.is_some());

        let entity = &ent.unwrap();
        assert_eq!("foobar", &entity.class_name);
        assert!(entity.origin.length() > 4.0);
        assert!(entity.properties.contains_key("bar"));
        assert_eq!("baz", entity.properties.get("bar").unwrap());
    }

    #[test]
    fn test_broken()
    {
        assert!(parse_entity(&mut "".chars()).is_none());
        assert!(parse_entity(&mut "}".chars()).is_none());
        assert!(parse_entity(&mut "{".chars()).is_some());  // Just an opening bracket should result in an empty entity

        // Broken off origin tag should still be accepted and result in a default origin vector
        let ent = parse_entity(&mut "{ classname xyz origin".chars());
        assert!(ent.is_some());
        let entity = &ent.unwrap();
        assert_eq!("xyz", &entity.class_name);
        assert_eq!(0.0, entity.origin.length());
    }
}
