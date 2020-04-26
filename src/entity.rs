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

pub fn parse_entities(string: &String) -> Vec<Entity>
{
    let mut entities = Vec::new();
    let mut chars = string.chars();

    loop
    {
        loop
        {
            match next_token(&mut chars)
            {
                Some(token) if token == "{" => break,
                Some(_) => continue,
                None => return entities,
            }
        }

        let mut entity = Entity::default();
        loop
        {
            match next_token(&mut chars)
            {
                Some(token) if token == "}" => break,
                Some(key) => { entity.properties.insert(key, next_token(&mut chars).unwrap_or_default()); },
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

        entities.push(entity);
    }
}
