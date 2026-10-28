//! matrix-calc: Matris toplama, çarpma, transpoz, determinant ve iz hesaplama.
//!
//! Kullanım:
//!     matrix-calc add "1,2;3,4" "5,6;7,8"
//!     matrix-calc mul "1,2;3,4" "5,6;7,8"
//!     matrix-calc transpose "1,2,3;4,5,6"
//!     matrix-calc det "1,2;3,4"
//!     matrix-calc trace "1,2;3,4"
use std::env;
use std::fmt;
use std::process;

type Matrix = Vec<Vec<f64>>;

fn parse_matrix(s: &str) -> Result<Matrix, String> {
    let rows: Result<Vec<Vec<f64>>, _> = s.split(';')
        .map(|row| row.split(',').map(|v| v.trim().parse::<f64>()).collect())
        .collect();
    let m = rows.map_err(|_| format!("Geçersiz matris: '{}'", s))?;
    if m.is_empty() { return Err("Boş matris".into()); }
    let cols = m[0].len();
    if m.iter().any(|r| r.len() != cols) { return Err("Satır uzunlukları eşit değil".into()); }
    Ok(m)
}

struct MatDisplay<'a>(&'a Matrix);
impl fmt::Display for MatDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row in self.0 {
            let parts: Vec<String> = row.iter().map(|v| format!("{:8.3}", v)).collect();
            writeln!(f, "  [{}]", parts.join(", "))?;
        }
        Ok(())
    }
}

fn add(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    if a.len()!=b.len() || a[0].len()!=b[0].len() { return Err("Boyutlar eşleşmiyor".into()); }
    Ok(a.iter().zip(b.iter()).map(|(ra,rb)| ra.iter().zip(rb.iter()).map(|(x,y)| x+y).collect()).collect())
}

fn mul(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    let (m, k1, k2, n) = (a.len(), a[0].len(), b.len(), b[0].len());
    if k1 != k2 { return Err(format!("{}x{} × {}x{} çarpılamaz", m, k1, k2, n)); }
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m { for j in 0..n { for k in 0..k1 { c[i][j] += a[i][k] * b[k][j]; } } }
    Ok(c)
}

fn transpose(a: &Matrix) -> Matrix {
    let (m, n) = (a.len(), a[0].len());
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

fn det(a: &Matrix) -> Result<f64, String> {
    let n = a.len();
    if a.iter().any(|r| r.len() != n) { return Err("Kare matris gerekli".into()); }
    if n == 1 { return Ok(a[0][0]); }
    if n == 2 { return Ok(a[0][0]*a[1][1] - a[0][1]*a[1][0]); }
    // LU yöntemi (basit)
    let mut m: Vec<Vec<f64>> = a.iter().map(|r| r.clone()).collect();
    let mut sign = 1.0f64;
    for col in 0..n {
        // Pivot bul
        let pivot = (col..n).max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap()).unwrap();
        if m[pivot][col].abs() < 1e-12 { return Ok(0.0); }
        if pivot != col { m.swap(col, pivot); sign = -sign; }
        let pv = m[col][col];
        for row in (col+1)..n {
            let factor = m[row][col] / pv;
            for c2 in col..n { let v = m[col][c2]; m[row][c2] -= factor * v; }
        }
    }
    Ok(sign * (0..n).map(|i| m[i][i]).product::<f64>())
}

fn trace(a: &Matrix) -> Result<f64, String> {
    let n = a.len();
    if a.iter().any(|r| r.len() != n) { return Err("Kare matris gerekli".into()); }
    Ok((0..n).map(|i| a[i][i]).sum())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        println!("Kullanım: matrix-calc <add|mul|transpose|det|trace> <matris1> [matris2]");
        println!("Matris formatı: \"satır1_e1,satır1_e2;satır2_e1,satır2_e2\"");
        println!("Örnek: matrix-calc det \"1,2;3,4\"");
        return;
    }
    match args[0].as_str() {
        "add" | "mul" => {
            if args.len() < 3 { eprintln!("İki matris gerekli."); process::exit(1); }
            let a = parse_matrix(&args[1]).unwrap_or_else(|e| { eprintln!("{}", e); process::exit(1); });
            let b = parse_matrix(&args[2]).unwrap_or_else(|e| { eprintln!("{}", e); process::exit(1); });
            let r = if args[0]=="add" { add(&a,&b) } else { mul(&a,&b) }
                .unwrap_or_else(|e| { eprintln!("{}", e); process::exit(1); });
            println!("Sonuç:\n{}", MatDisplay(&r));
        }
        "transpose" => {
            if args.len() < 2 { eprintln!("Matris gerekli."); process::exit(1); }
            let a = parse_matrix(&args[1]).unwrap_or_else(|e| { eprintln!("{}", e); process::exit(1); });
            println!("Transpoz:\n{}", MatDisplay(&transpose(&a)));
        }
        "det" => {
            if args.len() < 2 { eprintln!("Matris gerekli."); process::exit(1); }
            let a = parse_matrix(&args[1]).unwrap_or_else(|e| { eprintln!("{}", e); process::exit(1); });
            match det(&a) {
                Ok(d) => println!("Determinant: {:.6}", d),
                Err(e) => { eprintln!("{}", e); process::exit(1); }
            }
        }
        "trace" => {
            if args.len() < 2 { eprintln!("Matris gerekli."); process::exit(1); }
            let a = parse_matrix(&args[1]).unwrap_or_else(|e| { eprintln!("{}", e); process::exit(1); });
            match trace(&a) {
                Ok(t) => println!("İz (Trace): {:.6}", t),
                Err(e) => { eprintln!("{}", e); process::exit(1); }
            }
        }
        _ => { eprintln!("Geçersiz komut."); process::exit(1); }
    }
}
