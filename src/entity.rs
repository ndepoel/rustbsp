use std::str::Chars;
use std::collections::HashMap;

use cgmath::{ Vector3, Zero };

use super::parser;

#[derive(Debug)]
pub struct Entity
{
    pub class_name: String,
    pub origin: Vector3<f32>,
    pub angle: f32,
    pub properties: HashMap<String, String>,
}

impl Entity
{
    pub fn default() -> Self
    {
        Self { class_name: String::default(), origin: Vector3::<f32>::zero(), angle: f32::default(), properties: HashMap::<String, String>::default() }
    }

    pub fn get_vector(&self, prop_name: &str) -> Option<Vector3::<f32>>
    {
        self.properties.get(prop_name).and_then(|p| Some(parser::parse_vector(p)))
    }
}

fn parse_entity(chars: &mut Chars<'_>) -> Option<Entity>
{
    loop
    {
        match parser::next_token(chars)
        {
            Some(token) if token == "{" => break,
            Some(_) => continue,
            None => return None,
        }
    }

    let mut entity = Entity::default();
    loop
    {
        match parser::next_token(chars)
        {
            Some(token) if token == "}" => break,
            Some(key) => { entity.properties.insert(key, parser::next_token(chars).unwrap_or_default()); },
            None => break,
        }
    }

    if let Some(class_name) = entity.properties.remove("classname")
    {
        entity.class_name = class_name.clone();
    }

    if let Some(origin) = entity.properties.remove("origin")
    {
        entity.origin = parser::parse_vector(&origin);
    }

    if let Some(angle) = entity.properties.remove("angle")
    {
        entity.angle = angle.parse::<f32>().unwrap_or_default();
    }

    Some(entity)
}

pub fn parse_entities(string: &str) -> Vec<Entity>
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
        assert_eq!(Vector3::new(1.1, 2.2, 3.3), entity.origin);
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
        assert_eq!(Vector3::<f32>::zero(), entity.origin);
    }
}
