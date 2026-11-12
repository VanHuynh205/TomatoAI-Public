export {};

declare global {
  interface Window {
    pywebview?: {
      api: {
        get_session_token(): Promise<string>;
        open_external_url(url: string): Promise<void>;
      };
    };
  }
}
