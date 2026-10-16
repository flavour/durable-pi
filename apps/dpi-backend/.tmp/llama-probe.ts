import { createModels } from "@earendil-works/pi-ai/models";
import { installLlamaCpp, llamaModels, LLAMA_PROVIDER } from "../src/llamacpp.js";
const models = createModels();
console.log("base", installLlamaCpp(models, { url: "http://forge:9931/v1" }));
const list = await llamaModels({ url: "http://forge:9931" }, AbortSignal.timeout(5000));
console.log(JSON.stringify(list, null, 1));
await models.refresh({ providers: [LLAMA_PROVIDER], force: true });
console.log("available:", (await models.getAvailable(LLAMA_PROVIDER)).map((m) => `${m.provider}/${m.id} ctx=${m.contextWindow}`));
console.log("auth:", await models.getAuth(models.getModel(LLAMA_PROVIDER, list[0].id)!));
