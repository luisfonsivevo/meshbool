#[cfg(feature = "zngur")]
use zngur::Zngur;

fn main() {
	build_rs::output::rerun_if_changed("Cargo.toml");
	#[allow(unused)]
	let crate_dir = build_rs::input::cargo_manifest_dir();
	#[allow(unused)]
	let out_dir = build_rs::input::out_dir();

	#[cfg(feature = "zngur")]
	{
		build_rs::output::rerun_if_changed("main.zng");

		let generated_cpp = std::path::PathBuf::from("generated/meshbool/");
		let _ = std::fs::create_dir_all(&generated_cpp);
		let rs_file = out_dir.join("generated.rs");
		let h_file = generated_cpp.join("meshbool.h");

		Zngur::from_zng_file(crate_dir.join("main.zng"))
			.with_cpp_file(generated_cpp.join("generated.cpp"))
			.with_h_file(h_file)
			.with_rs_file(rs_file.clone())
			.generate();
		let s = std::fs::read_to_string(&rs_file).expect("File should exist");
		let new = s.replace("#[no_mangle]", "#[unsafe(no_mangle)]");
		std::fs::write(rs_file, new).expect("Failed to write file");
	}
}
