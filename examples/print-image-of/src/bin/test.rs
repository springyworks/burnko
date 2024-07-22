//rust

fn main() {
    let x = 10;
    let y = 20;

    println!("{x}{y}");
    println!("Image buffer created successfully! x = {}, y = {}", x, y);
    let number: f64 = 1.0;
    let width: usize = 5;
    println!("{number:>width$}");

    let f = 255.555555;
    let dec = 3;
    let width = 10;
    println!("{f} to {dec} decimal places is {f:-^width$.dec$} very cool!");
}
