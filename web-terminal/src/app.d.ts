declare module '*.css';

declare global {
  namespace App {
    // interface Error {}
    // interface Locals {}
    // interface PageData {}
    // interface PageState {}
    // interface Platform {}
  }

  interface Window {
    getScreenDetails?: () => Promise<{
      screens: Array<{
        availLeft: number;
        availTop: number;
        availWidth: number;
        availHeight: number;
        label?: string;
      }>;
    }>;
    documentPictureInPicture?: {
      requestWindow: (options?: { width?: number; height?: number }) => Promise<Window>;
    };
  }
}

export {};
