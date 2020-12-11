export interface Config {
  port: number;
}

export const config: Config = {
  port: parseInt(process.env.NODE_PORT, 10) || 8880,
};
