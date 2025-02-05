import { defineConfig } from "vite";
import wasmPack from "vite-plugin-wasm-pack";

export default defineConfig({
    server: {
        host: '0.0.0.0'
    },
    build: {
        minify: false,
    },
    plugins: [wasmPack(["mlp"])],
});
