use std::str::Chars;

use cgmath::{ Vector3, Zero };

pub fn next_token(chars: &mut Chars<'_>) -> Option<String>
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

    // Skip comment lines and look for the next token
    if token.starts_with("//")
    {
        skip_line(chars);
        return next_token(chars);
    }

    if token.len() > 0 { Some(token) } else { None }
}

fn skip_line(chars: &mut Chars<'_>)
{
    loop
    {
        let item = chars.next();
        match item
        {
            Some(c) if c == '\r' || c == '\n' => return,
            _ => continue,
        }
    }
}

pub fn tokenize(string: &str) -> Vec<String>
{
    let mut tokens = Vec::new();

    let mut chars = string.chars();
    while let Some(token) = next_token(&mut chars)
    {
        tokens.push(token);
    }

    tokens
}

pub fn parse_vector(string: &str) -> Vector3<f32>
{
    let mut chars = string.chars();
    Vector3::new(
        next_token(&mut chars).unwrap_or_default().parse::<f32>().unwrap_or_default(),
        next_token(&mut chars).unwrap_or_default().parse::<f32>().unwrap_or_default(),
        next_token(&mut chars).unwrap_or_default().parse::<f32>().unwrap_or_default(),
    )
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
        let vec = parse_vector("1 2.0 3.14");
        assert_eq!(1.0, vec.x);
        assert_eq!(2.0, vec.y);
        assert_eq!(3.14, vec.z);

        // Missing tokens should result in a default component value
        let vec = parse_vector("4 5");
        assert_eq!(4.0, vec.x);
        assert_eq!(5.0, vec.y);
        assert_eq!(0.0, vec.z);

        // Empty strings should be accepted, result in a default vector
        let vec = parse_vector("");
        assert_eq!(0.0, vec.x);
        assert_eq!(0.0, vec.y);
        assert_eq!(0.0, vec.z);

        // Test parse errors, invalid numbers should result in default values
        let vec = parse_vector("a b c");
        assert_eq!(0.0, vec.x);
        assert_eq!(0.0, vec.y);
        assert_eq!(0.0, vec.z);
    }
}
